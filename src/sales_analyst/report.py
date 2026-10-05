"""Сборка итогового отчёта в Markdown. Все таблицы и цифры рисует код из метрик."""
from __future__ import annotations

from datetime import datetime

from .agents.analyst import CHECKLIST, call_aggregates, money
from .models import AnalysisPlan, CallAnalysis, CallRecord, CrmSnapshot, Verification

METRIC_LABELS = {
    "leads_created": "новых сделок",
    "leads_created_prev": "новых сделок в прошлом периоде",
    "won_count": "выиграно сделок",
    "won_count_prev": "выиграно в прошлом периоде",
    "won_amount": "выручка",
    "won_amount_prev": "выручка в прошлом периоде",
    "lost_count": "проиграно сделок",
    "win_rate": "доля выигранных",
    "open_leads": "открытых сделок",
    "leads_without_tasks": "сделок без задач",
    "leads_overdue": "сделок с просроченной задачей",
    "tasks_completed": "выполнено задач",
    "calls_total": "звонков",
    "calls_talk": "разговоров",
    "talk_minutes": "минут разговоров",
    "calls_analyzed": "разобрано звонков",
    "avg_call_score": "средний балл звонков",
}
CHECK_LABELS = {
    "greeting": "Приветствие",
    "needs_discovery": "Потребность",
    "presentation": "Презентация",
    "objection_handling": "Возражения",
    "next_step": "След. шаг",
}
SEVERITY = {"high": "🔴", "medium": "🟠", "low": "⚪"}


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def _delta(cur: int, prev: int) -> str:
    return f"{cur} ({prev})"


class ReportBuilder:
    def __init__(self, snapshot: CrmSnapshot, crm_url: str | None = None) -> None:
        self.s = snapshot
        self.crm_url = crm_url
        self.managers = {m.user_id: m for m in snapshot.managers}
        self.calls: dict[int, CallAnalysis] = {}

    def lead(self, lead_id: int) -> str:
        name = self.s.lead_names.get(lead_id, f"#{lead_id}")
        if self.crm_url:
            return f"[{name}]({self.crm_url}/leads/detail/{lead_id})"
        return f"{name} (#{lead_id})"

    def ref(self, ref: str) -> str:
        kind, _, value = ref.partition(":")
        if kind == "lead":
            return self.lead(int(value))
        if kind == "call":
            a = self.calls.get(int(value))
            if a is None:
                return f"звонок #{value}"
            who = self.managers[a.manager_id].name if a.manager_id in self.managers else ""
            lead = f", {self.lead(a.lead_id)}" if a.lead_id else ""
            return f"звонок #{value} ({who}{lead})"
        if kind == "manager":
            return self.managers[int(value)].name
        if kind == "metric":
            owner, _, field = value.partition(".")
            m = self.s.department if owner in ("dept", "0") else self.managers[int(owner)]
            raw = getattr(m, field)
            val = "—" if raw is None else _pct(raw) if field == "win_rate" else money(raw) if "amount" in field else raw
            return f"{m.name}: {METRIC_LABELS.get(field, field)} — {val}"
        return ref

    def build(
        self,
        request: str,
        plan: AnalysisPlan,
        verification: Verification,
        calls: list[CallRecord],
        analyses: list[CallAnalysis],
        mode: str,
        run_id: str,
    ) -> str:
        s, d = self.s, self.s.department
        self.calls = {a.call_id: a for a in analyses}
        out: list[str] = []
        add = out.append

        add(f"# Работа отдела продаж за {s.period.days} дн. ({s.period.label()})\n")
        add(f"> Запрос: «{request}»  ")
        add(f"> Сформирован {datetime.now():%d.%m.%Y %H:%M} · режим: {mode} · прогон `{run_id}`\n")
        if plan.focus:
            add(f"**Акцент запроса:** {plan.focus}\n")

        add("## Главное\n")
        add(verification.headline + "\n")

        add("## Показатели\n")
        add("В скобках — предыдущий такой же период.\n")
        add("| Менеджер | Новые сделки | Выиграно | Выручка | Проиграно | Доля выигр. | Открыто | Без задачи | Просрочено | Звонки / разговоры | Балл звонков |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in [*s.managers, d]:
            name = f"**{m.name}**" if m.user_id == 0 else m.name
            score = "—" if m.avg_call_score is None else f"{m.avg_call_score:.0f} ({m.calls_analyzed})"
            add(
                f"| {name} | {_delta(m.leads_created, m.leads_created_prev)} | {_delta(m.won_count, m.won_count_prev)} "
                f"| {money(m.won_amount)} ({money(m.won_amount_prev)}) | {m.lost_count} | {_pct(m.win_rate)} "
                f"| {m.open_leads} | {m.leads_without_tasks} | {m.leads_overdue} | {m.calls_total} / {m.calls_talk} | {score} |"
            )
        add("")

        add("## Что происходит и почему\n")
        if not verification.accepted_findings:
            add("Существенных отклонений не найдено.\n")
        for f in verification.accepted_findings:
            add(f"### {SEVERITY[f.severity]} {f.title}\n")
            add(f"{f.explanation}\n")
            add("Основание: " + "; ".join(self.ref(r) for r in f.evidence) + ".\n")

        add("## Проблемные сделки\n")
        if s.problem_leads:
            add(f"Всего {len(s.problem_leads)}: без открытых задач — {d.leads_without_tasks}, "
                f"с просроченной задачей — {d.leads_overdue}. Первые 15 по сумме:\n")
            add("| Сделка | Менеджер | Этап | Сумма | Проблема | Без изменений, дн. |")
            add("|---|---|---|---:|---|---:|")
            for p in s.problem_leads[:15]:
                problem = "нет задачи" if p.reason == "no_task" else f"просрочка {p.overdue_days} дн."
                add(f"| {self.lead(p.lead_id)} | {p.manager} | {p.stage} | {money(p.price)} | {problem} | {p.days_since_update} |")
            add("")
        else:
            add("Все открытые сделки под контролем: у каждой есть задача в срок.\n")

        add("## Причины отказов\n")
        if s.loss_reasons:
            add("**По CRM** (поле «Причина отказа»): " + ", ".join(f"{k} — {v}" for k, v in s.loss_reasons.items()) + ".\n")
        said = [(a.call_id, a.refusal_reason) for a in analyses if a.refusal_reason]
        if said:
            add("**Со слов клиентов в звонках:**\n")
            add("\n".join(f"- «{reason}» — {self.ref(f'call:{cid}')}" for cid, reason in said) + "\n")
        if not s.loss_reasons and not said:
            add("Данных об отказах за период нет.\n")

        add("## Качество звонков\n")
        aggregates = call_aggregates(analyses)
        if aggregates:
            talks = sum(m.calls_talk for m in s.managers)
            add(f"Разобрано {len(analyses)} из {talks} разговоров длиннее порога. Балл 0–100 по чек-листу; "
                "в колонках чек-листа — среднее от 0 до 2.\n")
            add("| Менеджер | Звонков | Балл | " + " | ".join(CHECK_LABELS.values()) + " | Частые ошибки |")
            add("|---|---:|---:|" + "---:|" * len(CHECKLIST) + "---|")
            for uid, agg in aggregates.items():
                name = self.managers[uid].name if uid in self.managers else str(uid)
                checks = " | ".join(f"{agg['checklist_avg_0_2'][k]:.1f}" for k in CHECKLIST)
                add(f"| {name} | {agg['calls_analyzed']} | {agg['avg_score']:.0f} | {checks} | {'; '.join(agg['top_mistakes']) or '—'} |")
            add("")
        else:
            add("Звонки за период не анализировались.\n")

        add("## Рекомендации\n")
        if not verification.recommendations:
            add("Нет.")
        for i, r in enumerate(verification.recommendations, 1):
            add(f"{i}. {r.action} — *{r.owner}*")
        add("")

        add("## Требует решения человека\n")
        add("Система ничего не меняет в CRM и не делает выводов о сотрудниках сама. Ниже — то, где нужен руководитель.\n")
        for item in verification.human_review:
            if item.kind == "data":
                add(f"- **Данные:** {item.reason}")
            elif item.kind == "finding":
                add(f"- **{item.ref}** — {item.reason}")
            else:
                ref = self.ref(item.ref) if ":" in item.ref else item.ref
                add(f"- **{ref}** — {item.reason}")
        add("")

        add("## Как получен отчёт\n")
        add(
            f"- amoCRM: сделки, созданные и закрытые за период и за предыдущий период, все открытые сделки, "
            f"задачи; сделок в выборке: {len(s.lead_names)}.\n"
            f"- Телефония: звонков в примечаниях amoCRM — {len(calls)}, разобрано — {len(analyses)} "
            f"(запись → распознавание → оценка по чек-листу).\n"
            f"- Выводы аналитика проверены по ссылкам на данные: принято {len(verification.accepted_findings)}, "
            f"отброшено {len(verification.rejected_findings)}."
        )
        return "\n".join(out) + "\n"
