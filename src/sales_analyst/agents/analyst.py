"""AI Business Analyst: метрики CRM + оценки звонков -> выводы «что происходит, почему, что делать».

LLM получает только уже посчитанные факты (JSON) и обязана ссылаться на них через evidence.
Цифры в таблицах отчёта рисует код, LLM пишет интерпретацию.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean

from ..llm import StructuredLLM, prompt
from ..models import AnalysisPlan, AnalystOutput, CallAnalysis, CrmSnapshot, Finding, Recommendation

CHECKLIST = ("greeting", "needs_discovery", "presentation", "objection_handling", "next_step")


def money(value: int) -> str:
    return f"{value:,} ₽".replace(",", " ")


def call_aggregates(analyses: list[CallAnalysis]) -> dict[int, dict]:
    by_manager: dict[int, list[CallAnalysis]] = defaultdict(list)
    for a in analyses:
        by_manager[a.manager_id].append(a)
    result = {}
    for uid, items in by_manager.items():
        result[uid] = {
            "calls_analyzed": len(items),
            "avg_score": round(mean(a.score for a in items), 1),
            "checklist_avg_0_2": {k: round(mean(getattr(a, k) for a in items), 2) for k in CHECKLIST},
            "outcomes": dict(Counter(a.outcome for a in items)),
            "top_mistakes": [m for m, _ in Counter(x for a in items for x in a.mistakes).most_common(3)],
            "refusal_reasons": [a.refusal_reason for a in items if a.refusal_reason],
        }
    return result


def build_facts(snapshot: CrmSnapshot, analyses: list[CallAnalysis], plan: AnalysisPlan) -> dict:
    aggregates = call_aggregates(analyses)
    def metrics(m) -> dict:
        data = m.model_dump()
        data["leads_no_task_or_overdue"] = m.leads_without_tasks + m.leads_overdue  # производное, для текста
        return data

    dept = metrics(snapshot.department)
    dept["user_id"] = "dept"
    return {
        "period": {"label": snapshot.period.label(), "days": snapshot.period.days},
        "focus": plan.focus or "общий обзор работы отдела",
        "department": dept,
        "managers": [{**metrics(m), "calls": aggregates.get(m.user_id, {})} for m in snapshot.managers],
        "crm_loss_reasons": snapshot.loss_reasons,
        "open_leads_by_stage": snapshot.open_by_stage,
        "problem_leads_top": [p.model_dump() for p in snapshot.problem_leads[:30]],
        "problem_leads_total": len(snapshot.problem_leads),
        "calls": [
            {
                "call_id": a.call_id,
                "manager_id": a.manager_id,
                "lead_id": a.lead_id,
                "score": a.score,
                "outcome": a.outcome,
                "refusal_reason": a.refusal_reason,
                "mistakes": a.mistakes,
                "summary": a.summary,
            }
            for a in sorted(analyses, key=lambda a: a.score)[:40]
        ],
    }


def analyze_with_llm(facts: dict, llm: StructuredLLM) -> AnalystOutput:
    user = "<facts>\n" + json.dumps(facts, ensure_ascii=False, indent=1) + "\n</facts>"
    return llm.invoke(AnalystOutput, prompt("analyst"), user, smart=True)


def analyze_rule_based(snapshot: CrmSnapshot, analyses: list[CallAnalysis]) -> AnalystOutput:
    """Офлайн-вариант без LLM: пороговые правила. Те же структуры, тот же верификатор."""
    findings: list[Finding] = []
    recs: list[Recommendation] = []
    dept = snapshot.department
    aggregates = call_aggregates(analyses)

    if dept.leads_created_prev and dept.leads_created < dept.leads_created_prev * 0.85:
        findings.append(Finding(
            title="Входящий поток сделок снизился",
            explanation=f"Новых сделок {dept.leads_created} против {dept.leads_created_prev} в прошлом периоде.",
            severity="medium",
            evidence=["metric:dept.leads_created", "metric:dept.leads_created_prev"],
        ))

    for m in snapshot.managers:
        hygiene = m.leads_without_tasks + m.leads_overdue
        if m.open_leads and hygiene / m.open_leads >= 0.3:
            findings.append(Finding(
                title=f"{m.name}: сделки без контроля",
                explanation=(
                    f"Из {m.open_leads} открытых сделок {m.leads_without_tasks} без задачи и {m.leads_overdue} "
                    "с просроченной задачей — по ним никто не ведёт клиента дальше."
                ),
                severity="high" if hygiene / m.open_leads >= 0.5 else "medium",
                evidence=[f"metric:{m.user_id}.leads_without_tasks", f"metric:{m.user_id}.leads_overdue"],
            ))
            recs.append(Recommendation(
                action="Разобрать сделки без задач: поставить задачу с датой или закрыть с причиной",
                owner=m.name,
                evidence=[f"manager:{m.user_id}"],
            ))
        agg = aggregates.get(m.user_id)
        if agg and agg["checklist_avg_0_2"]["next_step"] < 1:
            findings.append(Finding(
                title=f"{m.name}: звонки заканчиваются без следующего шага",
                explanation="В большинстве разобранных звонков нет договорённости о конкретном следующем шаге.",
                severity="medium",
                evidence=[f"metric:{m.user_id}.avg_call_score"],
            ))

    if snapshot.loss_reasons:
        reason, count = next(iter(snapshot.loss_reasons.items()))
        findings.append(Finding(
            title=f"Главная причина отказов в CRM: «{reason}»",
            explanation=f"{count} из {dept.lost_count} проигранных сделок закрыты с этой причиной.",
            severity="medium",
            evidence=["metric:dept.lost_count"],
        ))

    no_next_step = [m.name for m in snapshot.managers if (a := aggregates.get(m.user_id)) and a["checklist_avg_0_2"]["next_step"] < 1]
    if no_next_step:
        recs.append(Recommendation(
            action=f"Разобрать на планёрке звонки ({', '.join(no_next_step)}): закрывать разговор на конкретный шаг с датой",
            owner="руководитель",
            evidence=[f"manager:{m.user_id}" for m in snapshot.managers if m.name in no_next_step],
        ))
    return AnalystOutput(headline=code_headline(snapshot), findings=findings, recommendations=recs)


def code_headline(snapshot: CrmSnapshot) -> str:
    """Сводка из чистых метрик — запасной вариант, если в сводке LLM нашлись неподтверждённые числа."""
    d = snapshot.department
    return (
        f"Новых сделок — {d.leads_created} (в прошлом периоде {d.leads_created_prev}), выиграно — "
        f"{d.won_count} на {money(d.won_amount)}. Открытых сделок без задачи или с просроченной задачей — "
        f"{d.leads_without_tasks + d.leads_overdue} из {d.open_leads}."
    )
