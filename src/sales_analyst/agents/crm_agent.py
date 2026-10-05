"""AI CRM Agent: сделки и задачи из amoCRM -> метрики по менеджерам и проблемные сделки.

Агент детерминированный: всё, что можно посчитать, считает код, а не LLM. Это дешевле,
воспроизводимо и исключает «галлюцинации» в цифрах отчёта.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..models import AnalysisPlan, CallAnalysis, CallRecord, CrmSnapshot, ManagerMetrics, Period, ProblemLead
from ..sources.amocrm import CLOSED_STATUSES, LOST_STATUS, WON_STATUS, AmoCrmClient, AmoCrmError

log = logging.getLogger(__name__)
DAY = 86400


@dataclass
class CrmContext:
    """Снимок для отчёта + служебные данные для Call Quality Agent."""

    snapshot: CrmSnapshot
    contact_to_lead: dict[int, int] = field(default_factory=dict)
    all_manager_ids: set[int] = field(default_factory=set)  # весь отдел: для строки «Весь отдел» и звонков
    manager_ids: set[int] = field(default_factory=set)  # кого показываем построчно (фильтр из запроса)
    warnings: list[str] = field(default_factory=list)


def find_problem_leads(
    open_leads: list[dict[str, Any]], open_tasks: list[dict[str, Any]], now: int
) -> list[tuple[dict[str, Any], str, int]]:
    """Сделки без открытых задач и сделки с просроченной задачей.

    Возвращает (сделка, причина, дней просрочки). Та же логика, что в snippets/overdue_leads.py.
    """
    tasks_by_lead: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in open_tasks:
        if t.get("entity_type", "leads") == "leads":
            tasks_by_lead[t["entity_id"]].append(t)

    result = []
    for lead in open_leads:
        if lead["status_id"] in CLOSED_STATUSES:
            continue
        tasks = tasks_by_lead.get(lead["id"], [])
        if not tasks:
            result.append((lead, "no_task", 0))
            continue
        oldest_deadline = min(t["complete_till"] for t in tasks)
        if oldest_deadline < now:
            result.append((lead, "overdue_task", (now - oldest_deadline) // DAY))
    return result


def name_matches(full_name: str, wanted: list[str]) -> bool:
    """Сравнение по словам с отсечением окончаний: «Петрову», «Петрова» -> «Игорь Петров»; «Анна» != «Жанна»."""
    for w in wanted:
        for part in full_name.lower().split():
            w_low = w.lower()
            k = max(4, min(len(part), len(w_low)) - 2)
            if len(part) >= 3 and len(w_low) >= 3 and part[:k] == w_low[:k]:
                return True
    return False


def collect_crm(client: AmoCrmClient, period: Period, plan: AnalysisPlan, now: int) -> CrmContext:
    prev = period.previous()
    warnings: list[str] = []
    try:
        users = {u["id"]: u["name"] for u in client.users()}
    except AmoCrmError as exc:  # /users доступен только администратору аккаунта
        log.warning("users: %s", exc)
        users = {}
        warnings.append("нет доступа к списку пользователей amoCRM (нужен токен администратора) — менеджеры показаны по ID")
    pipelines = client.pipelines()
    several = len(pipelines) > 1
    stage_names = {
        s["id"]: f"{p['name']} / {s['name']}" if several else s["name"]
        for p in pipelines for s in p.get("_embedded", {}).get("statuses", [])
    }

    created = client.leads_created(period.start, period.end)
    created_prev = client.leads_created(prev.start, prev.end)
    closed = client.leads_closed(period.start, period.end)
    closed_prev = client.leads_closed(prev.start, prev.end)
    open_leads = client.open_leads(pipelines)
    open_tasks = client.open_lead_tasks()
    completed_tasks = client.completed_tasks(period.start, period.end)

    # Отдел — все, кто отвечает за сделки текущего или прошлого периода.
    all_ids = {l["responsible_user_id"] for l in [*created, *created_prev, *closed, *closed_prev, *open_leads]}
    all_ids.discard(0)
    name = lambda uid: users.get(uid, f"Пользователь #{uid}")  # noqa: E731
    shown = set(all_ids)
    if plan.manager_names:
        shown = {uid for uid in all_ids if name_matches(name(uid), plan.manager_names)}
        if not shown:
            warnings.append(f"менеджеры {', '.join(plan.manager_names)} не найдены — показан весь отдел")
            shown = set(all_ids)

    metrics = {uid: ManagerMetrics(user_id=uid, name=name(uid)) for uid in all_ids}
    dept = ManagerMetrics(user_id=0, name="Весь отдел")

    def bump(lead: dict[str, Any], attr: str, value: int = 1) -> None:
        m = metrics.get(lead["responsible_user_id"])
        if m is not None:
            setattr(m, attr, getattr(m, attr) + value)
            setattr(dept, attr, getattr(dept, attr) + value)

    for l in created:
        bump(l, "leads_created")
    for l in created_prev:
        bump(l, "leads_created_prev")

    loss_reasons: Counter[str] = Counter()
    for l in closed:
        if l["status_id"] == WON_STATUS:
            bump(l, "won_count")
            bump(l, "won_amount", l.get("price") or 0)
        elif l["status_id"] == LOST_STATUS:
            bump(l, "lost_count")
            if l["responsible_user_id"] in shown:
                reasons = (l.get("_embedded") or {}).get("loss_reason") or []
                loss_reasons[reasons[0]["name"] if reasons else "причина не указана"] += 1
    for l in closed_prev:
        if l["status_id"] == WON_STATUS:
            bump(l, "won_count_prev")
            bump(l, "won_amount_prev", l.get("price") or 0)

    open_by_stage: Counter[str] = Counter()
    for l in open_leads:
        bump(l, "open_leads")
        if l["responsible_user_id"] in shown:
            open_by_stage[stage_names.get(l["status_id"], str(l["status_id"]))] += 1

    for t in completed_tasks:
        bump({"responsible_user_id": t["responsible_user_id"]}, "tasks_completed")

    problems: list[ProblemLead] = []
    for lead, reason, overdue_days in find_problem_leads(open_leads, open_tasks, now):
        uid = lead["responsible_user_id"]
        if uid not in metrics:
            continue
        bump(lead, "leads_without_tasks" if reason == "no_task" else "leads_overdue")
        if uid not in shown:
            continue
        problems.append(
            ProblemLead(
                lead_id=lead["id"],
                name=lead.get("name") or f"Сделка #{lead['id']}",
                manager_id=uid,
                manager=name(uid),
                price=lead.get("price") or 0,
                stage=stage_names.get(lead["status_id"], str(lead["status_id"])),
                reason=reason,  # type: ignore[arg-type]
                overdue_days=overdue_days,
                days_since_update=(now - lead.get("updated_at", now)) // DAY,
            )
        )
    problems.sort(key=lambda p: (-p.price, -p.overdue_days))

    for m in [*metrics.values(), dept]:
        decided = m.won_count + m.lost_count
        m.win_rate = round(m.won_count / decided, 3) if decided else None

    contact_to_lead: dict[int, int] = {}
    lead_names: dict[int, str] = {}
    # Открытые сделки перезаписывают закрытые: звонок по контакту скорее относится к живой сделке.
    for l in [*closed, *created, *open_leads]:
        lead_names[l["id"]] = l.get("name") or f"Сделка #{l['id']}"
        for c in (l.get("_embedded") or {}).get("contacts") or []:
            contact_to_lead[c["id"]] = l["id"]

    snapshot = CrmSnapshot(
        period=period,
        managers=sorted((metrics[uid] for uid in shown), key=lambda m: m.name),
        department=dept,
        problem_leads=problems,
        loss_reasons=dict(loss_reasons.most_common()),
        open_by_stage=dict(open_by_stage.most_common()),
        lead_names=lead_names,
    )
    return CrmContext(
        snapshot=snapshot,
        contact_to_lead=contact_to_lead,
        all_manager_ids=all_ids,
        manager_ids=shown,
        warnings=warnings,
    )


def with_call_metrics(
    snapshot: CrmSnapshot,
    calls: list[CallRecord],
    analyses: list[CallAnalysis],
    min_seconds: int,
    dept_scores: bool = True,
) -> CrmSnapshot:
    """Новый снимок с объёмом звонков и средним баллом качества (исходный не меняется)."""
    snap = snapshot.model_copy(deep=True)
    by_id = {m.user_id: m for m in snap.managers}
    dept = snap.department
    talk_seconds: Counter[int] = Counter()
    for c in calls:
        for key, m in ((c.manager_id, by_id.get(c.manager_id)), (0, dept)):
            if m is None:
                continue
            m.calls_total += 1
            if c.duration >= min_seconds:
                m.calls_talk += 1
                talk_seconds[key] += c.duration
    for key, seconds in talk_seconds.items():
        (dept if key == 0 else by_id[key]).talk_minutes = round(seconds / 60)
    scores: dict[int, list[int]] = defaultdict(list)
    for a in analyses:
        scores[a.manager_id].append(a.score)
        if dept_scores:  # при фильтре по менеджеру разобраны не все — средний по отделу был бы неверным
            scores[0].append(a.score)
    for uid, values in scores.items():
        m = dept if uid == 0 else by_id.get(uid)
        if m is not None:
            m.calls_analyzed = len(values)
            m.avg_call_score = round(sum(values) / len(values), 1)
    return snap
