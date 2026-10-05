from sales_analyst.agents import analyst, call_quality, crm_agent, supervisor
from sales_analyst.demo.transcripts import TRANSCRIPTS
from sales_analyst.models import AnalysisPlan, AnalystOutput, CallRecord, Finding, Period, Recommendation
from sales_analyst.sources.telephony import redact_pii

from .conftest import NOW

DAY = 86400


# ---------- CRM Agent ----------

def test_find_problem_leads():
    leads = [{"id": i, "status_id": 101} for i in (1, 2, 3)] + [{"id": 4, "status_id": 142}]
    tasks = [
        {"entity_id": 2, "entity_type": "leads", "complete_till": NOW - 3 * DAY},
        {"entity_id": 2, "entity_type": "leads", "complete_till": NOW + DAY},
        {"entity_id": 3, "entity_type": "leads", "complete_till": NOW + DAY},
    ]
    result = {(l["id"], reason, days) for l, reason, days in crm_agent.find_problem_leads(leads, tasks, NOW)}
    assert result == {(1, "no_task", 0), (2, "overdue_task", 3)}


def test_collect_crm_matches_dataset(crm, dataset):
    period = Period(start=NOW - 30 * DAY, end=NOW)
    ctx = crm_agent.collect_crm(crm, period, AnalysisPlan(), NOW)
    d = ctx.snapshot.department
    in_period = lambda ts: ts is not None and period.start <= ts < period.end  # noqa: E731
    assert d.leads_created == sum(in_period(l["created_at"]) for l in dataset.leads)
    assert d.won_count == sum(in_period(l["closed_at"]) and l["status_id"] == 142 for l in dataset.leads)
    assert d.open_leads == sum(l["status_id"] not in (142, 143) for l in dataset.leads)
    assert d.leads_without_tasks + d.leads_overdue == len(ctx.snapshot.problem_leads)
    assert sum(m.leads_created for m in ctx.snapshot.managers) == d.leads_created
    assert 1000 not in ctx.manager_ids  # администратор без сделок не попадает в отчёт


def test_collect_crm_filters_by_manager_name(crm):
    period = Period(start=NOW - 30 * DAY, end=NOW)
    ctx = crm_agent.collect_crm(crm, period, AnalysisPlan(manager_names=["Петров"]), NOW)
    assert [m.name for m in ctx.snapshot.managers] == ["Игорь Петров"]


# ---------- Call Quality Agent ----------

def _note(nid, entity_id, uniq, duration=120, uid=1):
    return {"id": nid, "entity_id": entity_id, "note_type": "call_out", "responsible_user_id": uid,
            "created_at": NOW, "params": {"uniq": uniq, "duration": duration, "link": f"http://r/{nid}"}}


def test_call_records_link_contacts_and_dedupe_by_uniq():
    notes = [("contacts", _note(1, 500, "u1")), ("leads", _note(2, 77, "u1")), ("contacts", _note(3, 999, "u2"))]
    calls = call_quality.build_call_records(notes, contact_to_lead={500: 77}, manager_ids={1})
    assert [(c.call_id, c.lead_id) for c in calls] == [(1, 77), (3, None)]


def test_selection_is_balanced_and_skips_short_calls():
    def rec(cid, uid, duration):
        return CallRecord(call_id=cid, lead_id=None, manager_id=uid, direction="out",
                          duration=duration, recording_url="x", created_at=cid)

    calls = [rec(i, 1, 120) for i in range(10)] + [rec(100 + i, 2, 120) for i in range(2)] + [rec(200, 2, 20)]
    selected = call_quality.select_for_analysis(calls, min_seconds=60, max_calls=4)
    assert sorted(c.manager_id for c in selected) == [1, 1, 2, 2]
    assert all(c.duration >= 60 for c in selected)


def test_heuristic_separates_good_and_bad_calls():
    good = call_quality.review_heuristic(TRANSCRIPTS["anna_1"])
    bad = call_quality.review_heuristic(TRANSCRIPTS["igor_2"])
    assert good.next_step == 2 and good.outcome == "следующий шаг согласован"
    assert bad.next_step == 0 and bad.outcome == "отказ"


def test_redact_pii():
    text = "Звоните +7 (999) 123-45-67 или 8 916 000 11 22, почта ivan@mail.ru, карта 2200 1234 5678 9012"
    assert redact_pii(text) == "Звоните [телефон] или [телефон], почта [email], карта [карта]"


# ---------- Supervisor ----------

def test_plan_offline():
    assert supervisor.plan_offline("Проанализируй работу отдела продаж за последние 30 дней").period_days == 30
    assert supervisor.plan_offline("Что было за неделю?").period_days == 7
    assert supervisor.plan_offline("Отчёт за 2 недели").period_days == 14
    assert supervisor.plan_offline("Цифры за квартал, без звонков").analyze_calls is False
    assert supervisor.plan_offline("Как работал Петров за 2 недели?").manager_names == ["Петров"]
    assert supervisor.plan_offline("Проанализируй работу отдела продаж за последние 30 дней").manager_names == []


def _snapshot(crm):
    return crm_agent.collect_crm(crm, Period(start=NOW - 30 * DAY, end=NOW), AnalysisPlan(), NOW).snapshot


def _verify(snap, findings, recs=(), headline="h"):
    output = AnalystOutput(headline=headline, findings=list(findings), recommendations=list(recs))
    facts = analyst.build_facts(snap, [], AnalysisPlan())
    return supervisor.verify(output, facts, snap, [], [], [], fallback_headline="CODE")


def test_verify_requires_all_refs_valid(crm):
    snap = _snapshot(crm)
    some_lead = next(iter(snap.lead_names))
    v = _verify(snap, [
        Finding(title="ok", explanation="", severity="low", evidence=[f"lead:{some_lead}", "manager:1001"]),
        Finding(title="отдел через 0", explanation="", severity="low", evidence=["metric:0.won_count"]),
        Finding(title="одна ссылка выдумана", explanation="", severity="low", evidence=["metric:dept.won_count", "lead:1"]),
        Finding(title="служебное поле", explanation="", severity="low", evidence=["metric:dept.name"]),
        Finding(title="без ссылок", explanation="", severity="high", evidence=[]),
    ], recs=[
        Recommendation(action="a", owner="b", evidence=["lead:1", "manager:1002"]),
        Recommendation(action="выдумка", owner="b", evidence=["lead:1"]),
        Recommendation(action="общая", owner="руководитель", evidence=[]),
    ])
    assert [f.title for f in v.accepted_findings] == ["ok", "отдел через 0"]
    assert v.accepted_findings[1].evidence == ["metric:dept.won_count"]
    assert [f.title for f in v.rejected_findings] == ["одна ссылка выдумана", "служебное поле", "без ссылок"]
    assert [(r.action, r.evidence) for r in v.recommendations] == [("a", ["manager:1002"]), ("общая", [])]
    assert sum("отброшен" in i.reason for i in v.human_review) == 4


def test_verify_checks_numbers_against_facts(crm):
    snap = _snapshot(crm)
    d = snap.department
    win_pct = round(d.win_rate * 100)
    v = _verify(snap, [
        Finding(title="Поток", explanation=f"Новых сделок {d.leads_created} против {d.leads_created_prev}, "
                f"доля выигранных {win_pct}%, выручка {d.won_amount:,} ₽".replace(",", " "),
                severity="low", evidence=["metric:dept.leads_created"]),
        Finding(title="Прогноз", explanation="Выручка упадёт на 4321 ₽", severity="high",
                evidence=["metric:dept.won_amount"]),
    ], headline="Выручка вырастет до 98765 ₽")
    assert [f.title for f in v.accepted_findings] == ["Поток"]
    assert v.headline == "CODE"  # сводка с выдуманным числом заменена сводкой из метрик


def test_name_matching_handles_cases():
    assert crm_agent.name_matches("Игорь Петров", ["Петрову"])
    assert crm_agent.name_matches("Мария Козлова", ["Козловой"])
    assert not crm_agent.name_matches("Жанна Иванова", ["Анна"])
    assert not crm_agent.name_matches("Анна Смирнова", ["Петров"])


def test_manager_filter_keeps_department_totals(crm):
    period = Period(start=NOW - 30 * DAY, end=NOW)
    full = crm_agent.collect_crm(crm, period, AnalysisPlan(), NOW).snapshot
    one = crm_agent.collect_crm(crm, period, AnalysisPlan(manager_names=["Петрова"]), NOW)
    assert [m.name for m in one.snapshot.managers] == ["Игорь Петров"]
    assert one.snapshot.department == full.department
    assert all(p.manager == "Игорь Петров" for p in one.snapshot.problem_leads)
    missing = crm_agent.collect_crm(crm, period, AnalysisPlan(manager_names=["Сидоренко"]), NOW)
    assert len(missing.snapshot.managers) == 4 and "не найдены" in missing.warnings[0]


def test_users_endpoint_forbidden_falls_back_to_ids(dataset):
    import httpx

    from sales_analyst.demo.fake_amocrm import BASE, make_transport
    from sales_analyst.sources.amocrm import AmoCrmClient

    inner = make_transport(dataset, fail_first=())

    def handler(request):
        if request.url.path.endswith("/users"):
            return httpx.Response(403)
        return inner.handle_request(request)

    client = AmoCrmClient(BASE, "demo-token", transport=httpx.MockTransport(handler), max_rps=1000)
    ctx = crm_agent.collect_crm(client, Period(start=NOW - 30 * DAY, end=NOW), AnalysisPlan(), NOW)
    assert ctx.snapshot.managers[0].name.startswith("Пользователь #")
    assert "администратора" in ctx.warnings[0]


def test_call_responsible_and_period_filter():
    note = _note(1, 500, "u1", uid=999)
    note["params"]["call_responsible"] = 1
    old = _note(2, 500, "u2")
    old["created_at"] = NOW - 40 * DAY
    calls = call_quality.build_call_records(
        [("contacts", note), ("contacts", old)], {500: 77}, {1}, Period(start=NOW - 30 * DAY, end=NOW + 1)
    )
    assert [(c.call_id, c.manager_id) for c in calls] == [(1, 1)]
