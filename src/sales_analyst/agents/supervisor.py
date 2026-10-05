"""AI Supervisor: разбирает запрос, проверяет выводы других агентов и решает, что отдать человеку.

Маршрутизацию между агентами делает граф LangGraph (graph.py), а здесь — две «умные» точки
супервизора: планирование и верификация.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator

from ..llm import StructuredLLM, prompt
from ..models import (
    AnalysisPlan,
    AnalystOutput,
    CallAnalysis,
    CallRecord,
    CrmSnapshot,
    HumanReviewItem,
    ManagerMetrics,
    Verification,
)
from .analyst import money

# ---------- планирование ----------

_UNITS = {"дн": 1, "ден": 1, "недел": 7, "месяц": 30, "квартал": 90, "год": 365}
_NOT_NAMES = {"Проанализируй", "Покажи", "Сравни", "Как", "Что", "Почему", "Отчёт", "Отчет", "Сделай", "Дай", "Цифры"}


def plan_offline(request: str) -> AnalysisPlan:
    """Разбор без LLM: период, «без звонков» и слова с заглавной буквы как кандидаты в фамилии.

    Если кандидат не совпадёт ни с одним менеджером, CRM Agent покажет весь отдел и предупредит.
    """
    text = request.lower()
    days = 30
    m = re.search(r"(\d+)\s*(дн|ден|недел|месяц|квартал|год)", text)
    if m:
        days = int(m.group(1)) * _UNITS[m.group(2)]
    else:
        for word, value in (("недел", 7), ("месяц", 30), ("квартал", 90), ("год", 365)):
            if re.search(rf"за\s+(последн\w+\s+)?{word}", text):
                days = value
                break
    names = [w for w in re.findall(r"\b[А-ЯЁ][а-яё]{2,}", request) if w not in _NOT_NAMES]
    return AnalysisPlan(
        period_days=min(days, 366),
        manager_names=names,
        analyze_calls=not re.search(r"без звонк|только crm|только цифр", text),
    )


def plan_request(request: str, llm: StructuredLLM | None) -> AnalysisPlan:
    if llm is None:
        return plan_offline(request)
    return llm.invoke(AnalysisPlan, prompt("planner"), f"<request>{request}</request>")


# ---------- верификация ----------

CITABLE_METRICS = set(ManagerMetrics.model_fields) - {"user_id", "name"}
LOW_CONFIDENCE = 0.6
_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b")
_NUMBER = re.compile(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?")


def normalize_ref(ref: str) -> str:
    """metric:0.<поле> — то же, что metric:dept.<поле> (в фактах у отдела user_id = 0)."""
    return "metric:dept." + ref.split(".", 1)[1] if ref.startswith("metric:0.") else ref


def ref_is_valid(ref: str, snapshot: CrmSnapshot, call_ids: set[int]) -> bool:
    kind, _, value = ref.partition(":")
    manager_ids = {m.user_id for m in snapshot.managers}
    if kind == "metric":
        owner, _, field = value.partition(".")
        if field not in CITABLE_METRICS:
            return False
        return owner == "dept" or (owner.isdigit() and int(owner) in manager_ids)
    if not value.isdigit():
        return False
    num = int(value)
    if kind == "lead":
        return num in snapshot.lead_names
    if kind == "call":
        return num in call_ids
    if kind == "manager":
        return num in manager_ids
    return False


def _walk_numbers(obj: object) -> Iterator[float]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_numbers(v)


def allowed_numbers(facts: dict) -> set[float]:
    """Все числа, которые модель вправе упомянуть: значения из фактов и их обычные записи (%, тыс., млн)."""
    out: set[float] = set()
    for v in _walk_numbers(facts):
        out |= {round(v, 1), float(round(v))}
        if 0 < v < 1:
            out |= {float(round(v * 100)), round(v * 100, 1)}
        if v >= 1_000:
            out |= {float(round(v / 1_000)), round(v / 1_000, 1)}
        if v >= 1_000_000:
            out |= {round(v / 1_000_000, 1), round(v / 1_000_000, 2)}
    return out


def unsupported_numbers(text: str, allowed: set[float]) -> list[str]:
    """Числа в тексте, которых нет в фактах. Даты и числа до 10 («2 звонка», «3 менеджера») не проверяем."""
    bad = []
    for raw in _NUMBER.findall(_DATE.sub(" ", text)):
        value = float(raw.replace(" ", "").replace(" ", "").replace(",", "."))
        if value > 10 and round(value, 1) not in allowed and round(value, 2) not in allowed:
            bad.append(raw)
    return bad


def verify(
    output: AnalystOutput,
    facts: dict,
    snapshot: CrmSnapshot,
    calls: list[CallRecord],
    analyses: list[CallAnalysis],
    warnings: list[str],
    fallback_headline: str,
) -> Verification:
    """Отсев неподтверждённых выводов + список того, что требует решения человека. Вход не меняется.

    Вывод принимается, только если (1) у него есть ссылки на факты, (2) все ссылки существуют,
    (3) каждое число в тексте вывода есть в фактах.
    """
    call_ids = {a.call_id for a in analyses}
    allowed = allowed_numbers(facts)
    v = Verification(headline=output.headline)

    bad_headline = unsupported_numbers(output.headline, allowed)
    if bad_headline:
        v.headline = fallback_headline
        v.notes.append(f"в сводке аналитика числа без подтверждения ({', '.join(bad_headline)}) — сводка собрана кодом")

    for f in output.findings:
        refs = [normalize_ref(r) for r in f.evidence]
        bad_refs = [r for r in refs if not ref_is_valid(r, snapshot, call_ids)]
        bad_nums = unsupported_numbers(f"{f.title} {f.explanation}", allowed)
        if refs and not bad_refs and not bad_nums:
            v.accepted_findings.append(f.model_copy(update={"evidence": refs}))
            continue
        v.invalid_refs += bad_refs
        v.rejected_findings.append(f)
        if not refs:
            why = "нет ссылок на данные"
        elif bad_refs:
            why = f"несуществующие ссылки: {', '.join(bad_refs)}"
        else:
            why = f"числа не из данных: {', '.join(bad_nums)}"
        v.notes.append(f"отброшен вывод «{f.title}»: {why}")

    for r in output.recommendations:
        refs = [normalize_ref(e) for e in r.evidence]
        good = [e for e in refs if ref_is_valid(e, snapshot, call_ids)]
        v.invalid_refs += [e for e in refs if e not in good]
        if refs and not good:
            v.notes.append(f"отброшена рекомендация «{r.action}»: ни одна ссылка не подтвердилась")
            continue
        v.recommendations.append(r.model_copy(update={"evidence": good}))

    # 1. Звонки, где модель сама не уверена, — прослушать человеку.
    for a in analyses:
        if a.method == "llm" and a.confidence < LOW_CONFIDENCE:
            v.human_review.append(HumanReviewItem(
                kind="call", ref=f"call:{a.call_id}",
                reason=f"оценка неуверенная ({a.confidence:.0%}) — прослушать вручную",
            ))
        elif a.outcome == "отказ" and a.score < 50:
            v.human_review.append(HumanReviewItem(
                kind="call", ref=f"call:{a.call_id}",
                reason=f"отказ клиента при балле {a.score}/100 — разобрать звонок с менеджером",
            ))
    # 2. Крупные сделки без контроля — решить руководителю: вернуть в работу или закрыть.
    for p in snapshot.problem_leads[:5]:
        what = "нет задачи" if p.reason == "no_task" else f"задача просрочена на {p.overdue_days} дн."
        v.human_review.append(HumanReviewItem(
            kind="lead", ref=f"lead:{p.lead_id}", reason=f"{money(p.price)}, {what} — вернуть в работу или закрыть",
        ))
    # 3. Выводы, влияющие на оценку людей, — только как сигнал, решение за руководителем.
    for f in v.accepted_findings:
        if f.severity == "high":
            v.human_review.append(HumanReviewItem(
                kind="finding", ref=f.title, reason="серьёзный вывод о работе сотрудника — подтвердить по исходным данным",
            ))
    # 4. Качество данных.
    if calls:
        unlinked = sum(1 for c in calls if c.lead_id is None)
        if unlinked / len(calls) > 0.2:
            v.human_review.append(HumanReviewItem(
                kind="data", ref="calls",
                reason=f"{unlinked} из {len(calls)} звонков не привязаны к сделке — проверить настройки виджета телефонии",
            ))
    if any(a.method == "heuristic" for a in analyses):
        v.human_review.append(HumanReviewItem(
            kind="data", ref="calls", reason="звонки оценены эвристикой, а не LLM — баллы ориентировочные",
        ))
    for w, n in Counter(warnings).items():
        v.human_review.append(HumanReviewItem(kind="data", ref="pipeline", reason=w if n == 1 else f"{w} (×{n})"))
    for note in v.notes:
        v.human_review.append(HumanReviewItem(kind="data", ref="analyst", reason=note))
    return v
