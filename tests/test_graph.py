"""Сквозные прогоны графа на демо-данных: офлайн, с фейковой LLM и с падающей LLM."""
from pydantic import BaseModel

from sales_analyst.graph import run
from sales_analyst.models import AnalysisPlan, AnalystOutput, CallReview, Finding, Recommendation

REQUEST = "Проанализируй работу отдела продаж за последние 30 дней"


class FakeLLM:
    """Возвращает правдоподобные structured outputs; аналитик «галлюцинирует» один вывод."""

    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    def invoke(self, schema: type[BaseModel], system: str, user: str, *, smart: bool = False):
        self.calls.append((schema.__name__, smart))
        if schema is AnalysisPlan:
            return AnalysisPlan(period_days=30)
        if schema is CallReview:
            assert "<transcript>" in user and "+7 999" not in user
            low = "не слышно" in user
            return CallReview(greeting=2, needs_discovery=1, presentation=1, objection_handling=2, next_step=1,
                              outcome="клиент думает", summary="s", confidence=0.3 if low else 0.9)
        if schema is AnalystOutput:
            assert "<facts>" in user
            return AnalystOutput(
                headline="Отдел держит выручку, но теряет контроль над сделками.",
                findings=[
                    Finding(title="Просрочки у Петрова", explanation="e", severity="high",
                            evidence=["metric:1002.leads_overdue"]),
                    Finding(title="Прогноз выручки упадёт на 40%", explanation="e", severity="high",
                            evidence=["metric:dept.revenue_forecast"]),
                    Finding(title="call: мало звонков у отдела", explanation="e", severity="high",
                            evidence=["metric:0.calls_total"]),
                ],
                recommendations=[Recommendation(action="Разобрать просрочки", owner="Игорь Петров",
                                                evidence=["manager:1002"])],
            )
        raise AssertionError(schema)


class BrokenLLM:
    def invoke(self, *args, **kwargs):
        raise TimeoutError("LLM недоступна")


def test_offline_run_produces_full_report(make_deps):
    deps = make_deps()
    state = run(REQUEST, deps)
    md = state["report"]
    for section in ("## Главное", "## Показатели", "## Что происходит и почему", "## Проблемные сделки",
                    "## Причины отказов", "## Качество звонков", "## Рекомендации", "## Требует решения человека"):
        assert section in md
    assert len(state["call_analyses"]) == 12  # все разговоры с записью, короткие звонки отсечены
    assert all(a.method == "heuristic" for a in state["call_analyses"])
    assert deps.storage.run_report(state["run_id"]) == md


def test_llm_run_drops_unsupported_findings(make_deps):
    llm = FakeLLM()
    state = run(REQUEST, make_deps(llm))
    v = state["verification"]
    assert [f.title for f in v.accepted_findings] == ["Просрочки у Петрова", "call: мало звонков у отдела"]
    assert [f.title for f in v.rejected_findings] == ["Прогноз выручки упадёт на 40%"]
    findings_section = state["report"].split("## Что происходит и почему")[1].split("## Проблемные сделки")[0]
    assert "Прогноз выручки" not in findings_section and "Просрочки у Петрова" in findings_section
    # обрывочный звонок с низкой уверенностью отдан человеку
    assert any(i.kind == "call" and "неуверенная" in i.reason for i in v.human_review)
    # дорогая модель вызывается один раз — только для итогового анализа
    assert [name for name, smart in llm.calls if smart] == ["AnalystOutput"]


def test_run_survives_llm_outage(make_deps):
    state = run(REQUEST, make_deps(BrokenLLM()))
    assert state["plan"].period_days == 30
    # звонки не потеряны — оценены эвристикой, предупреждение схлопнуто в одну строку
    assert len(state["call_analyses"]) == 12 and all(a.method == "heuristic" for a in state["call_analyses"])
    reasons = [i.reason for i in state["verification"].human_review]
    assert sum("LLM не ответила" in r for r in reasons) == 1 and any("(×12)" in r for r in reasons)
    assert any("правилами" in r for r in reasons)
    assert "## Показатели" in state["report"]


def test_run_without_stt_still_reports(make_deps):
    deps = make_deps()
    deps.transcriber = None
    state = run("Цифры за неделю", deps)
    assert state["call_analyses"] == [] and state["calls"]
    assert "не прослушаны" in state["report"]
    assert state["crm"].snapshot.department.calls_total == len(state["calls"])
