"""Граф агентов на LangGraph.

    START
      └─ supervisor_plan ──┬─ crm_agent ─────────┐
                           └─ telephony_fetch ───┴─ calls_prepare ─┬─ analyze_call ×N (Send) ─┐
                                                                    └──────────────────────────┴─ calls_done
    calls_done ─ business_analyst ─ supervisor_verify ─ report ─ END

Агенты общаются только через типизированное состояние. Ошибка одного звонка не роняет
прогон: она попадает в warnings и затем — в раздел «Требует решения человека».
"""
from __future__ import annotations

import logging
import operator
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import analyst, call_quality, crm_agent, supervisor
from .config import Settings
from .llm import StructuredLLM
from .models import AnalysisPlan, AnalystOutput, CallAnalysis, CallRecord, Period, Verification
from .report import ReportBuilder
from .sources.amocrm import AmoCrmClient
from .sources.telephony import RecordingFetcher, Transcriber
from .storage import Storage

log = logging.getLogger(__name__)


@dataclass
class Deps:
    settings: Settings
    crm: AmoCrmClient
    fetcher: RecordingFetcher
    transcriber: Transcriber | None  # None — STT не настроен, звонки только считаются
    storage: Storage
    llm: StructuredLLM | None
    crm_url: str | None = None
    now: int | None = None  # фиксируется в тестах


class State(TypedDict, total=False):
    request: str
    run_id: str
    now: int
    plan: AnalysisPlan
    period: Period
    crm: crm_agent.CrmContext
    raw_call_notes: list[call_quality.RawNote]
    calls: list[CallRecord]
    call_analyses: Annotated[list[CallAnalysis], operator.add]
    facts: dict
    analyst: AnalystOutput
    verification: Verification
    report: str
    report_path: str
    warnings: Annotated[list[str], operator.add]


def build_graph(deps: Deps):
    s = deps.settings

    def supervisor_plan(state: State) -> dict[str, Any]:
        warnings = []
        try:
            plan = supervisor.plan_request(state["request"], deps.llm)
        except Exception as exc:  # LLM недоступна — не падаем, разбираем запрос правилами
            log.exception("planner")
            plan = supervisor.plan_offline(state["request"])
            warnings.append(f"разбор запроса без LLM: {type(exc).__name__}")
        now = state["now"]
        period = Period(start=now - plan.period_days * 86400, end=now)
        deps.storage.update_run(state["run_id"], plan_json=plan.model_dump_json())
        return {"plan": plan, "period": period, "warnings": warnings}

    def crm(state: State) -> dict[str, Any]:
        ctx = crm_agent.collect_crm(deps.crm, state["period"], state["plan"], state["now"])
        return {"crm": ctx, "warnings": ctx.warnings}

    def telephony_fetch(state: State) -> dict[str, Any]:
        if not state["plan"].analyze_calls:
            return {"raw_call_notes": []}
        return {"raw_call_notes": call_quality.fetch_call_notes(deps.crm, state["period"])}

    def calls_prepare(state: State) -> dict[str, Any]:
        ctx = state["crm"]
        calls = call_quality.build_call_records(
            state["raw_call_notes"], ctx.contact_to_lead, ctx.all_manager_ids, state["period"]
        )
        warnings = []
        if state["plan"].analyze_calls and deps.transcriber is None and calls:
            warnings.append("распознавание речи не настроено (STT_BASE_URL) — звонки посчитаны, но не прослушаны")
        return {"calls": calls, "warnings": warnings}

    def route_calls(state: State):
        if deps.transcriber is None:
            return "calls_done"
        selected = call_quality.select_for_analysis(
            state["calls"], s.min_call_seconds, s.max_calls, only_managers=state["crm"].manager_ids
        )
        if not selected:
            return "calls_done"
        return [Send("analyze_call", {"call": c, "run_id": state["run_id"]}) for c in selected]

    def analyze_call(payload: dict[str, Any]) -> dict[str, Any]:
        call: CallRecord = payload["call"]
        assert deps.transcriber is not None
        try:
            a, warning = call_quality.analyze_call(call, deps.fetcher, deps.transcriber, deps.storage, deps.llm)
            deps.storage.save_call_analysis(payload["run_id"], a)
        except Exception as exc:  # битая ссылка, таймаут STT, ошибка БД — теряем один звонок, не прогон
            log.warning("call %s: %s", call.call_id, exc)
            return {"warnings": [f"звонок #{call.call_id} не разобран: {type(exc).__name__}"]}
        return {"call_analyses": [a], "warnings": [warning] if warning else []}

    def calls_done(state: State) -> dict[str, Any]:
        ctx = state["crm"]
        snap = crm_agent.with_call_metrics(
            ctx.snapshot,
            state.get("calls", []),
            state.get("call_analyses", []),
            s.min_call_seconds,
            dept_scores=ctx.manager_ids == ctx.all_manager_ids,
        )
        return {"crm": replace(ctx, snapshot=snap)}

    def business_analyst(state: State) -> dict[str, Any]:
        snap, analyses = state["crm"].snapshot, state.get("call_analyses", [])
        facts = analyst.build_facts(snap, analyses, state["plan"])
        if deps.llm is not None:
            try:
                return {"facts": facts, "analyst": analyst.analyze_with_llm(facts, deps.llm)}
            except Exception as exc:
                log.exception("analyst")
                return {
                    "facts": facts,
                    "analyst": analyst.analyze_rule_based(snap, analyses),
                    "warnings": [f"выводы построены правилами, LLM-аналитик недоступен: {type(exc).__name__}"],
                }
        return {"facts": facts, "analyst": analyst.analyze_rule_based(snap, analyses)}

    def supervisor_verify(state: State) -> dict[str, Any]:
        snap = state["crm"].snapshot
        v = supervisor.verify(
            state["analyst"],
            state["facts"],
            snap,
            state.get("calls", []),
            state.get("call_analyses", []),
            state.get("warnings", []),
            fallback_headline=analyst.code_headline(snap),
        )
        return {"verification": v}

    def report(state: State) -> dict[str, Any]:
        snap = state["crm"].snapshot
        mode = "LLM" if deps.llm is not None else "офлайн (без LLM)"
        md = ReportBuilder(snap, deps.crm_url).build(
            state["request"],
            state["plan"],
            state["verification"],
            state.get("calls", []),
            state.get("call_analyses", []),
            mode,
            state["run_id"],
        )
        path = Path(s.reports_dir) / f"{state['run_id']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        deps.storage.update_run(
            state["run_id"],
            metrics_json=snap.model_dump_json(),
            verification_json=state["verification"].model_dump_json(),
            report_md=md,
            status="done",
        )
        return {"report": md, "report_path": str(path)}

    g = StateGraph(State)
    g.add_node("supervisor_plan", supervisor_plan)
    g.add_node("crm_agent", crm)
    g.add_node("telephony_fetch", telephony_fetch)
    g.add_node("calls_prepare", calls_prepare)
    g.add_node("analyze_call", analyze_call)
    g.add_node("calls_done", calls_done)
    g.add_node("business_analyst", business_analyst)
    g.add_node("supervisor_verify", supervisor_verify)
    g.add_node("report", report)

    g.add_edge(START, "supervisor_plan")
    g.add_edge("supervisor_plan", "crm_agent")
    g.add_edge("supervisor_plan", "telephony_fetch")
    g.add_edge(["crm_agent", "telephony_fetch"], "calls_prepare")
    g.add_conditional_edges("calls_prepare", route_calls, ["analyze_call", "calls_done"])
    g.add_edge("analyze_call", "calls_done")
    g.add_edge("calls_done", "business_analyst")
    g.add_edge("business_analyst", "supervisor_verify")
    g.add_edge("supervisor_verify", "report")
    g.add_edge("report", END)
    return g.compile()


def run(request: str, deps: Deps) -> State:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    deps.storage.start_run(run_id, request)
    graph = build_graph(deps)
    try:
        return graph.invoke(
            {"request": request, "run_id": run_id, "now": deps.now or int(time.time()), "warnings": [], "call_analyses": []},
            config={"max_concurrency": 4},
        )
    except Exception:
        deps.storage.update_run(run_id, status="failed")
        raise
