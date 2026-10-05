"""Доменные модели. Всё, что передаётся между агентами, — типизированные pydantic-объекты."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------- План (Supervisor) ----------

class AnalysisPlan(BaseModel):
    """Разбор запроса руководителя в параметры прогона."""

    period_days: int = Field(30, ge=1, le=366, description="Длина анализируемого периода в днях")
    manager_names: list[str] = Field(
        default_factory=list, description="Фамилии/имена менеджеров, если запрос про конкретных людей; пусто = весь отдел"
    )
    analyze_calls: bool = Field(True, description="Нужно ли слушать звонки (false, если просили только CRM-цифры)")
    focus: str = Field("", description="На чём руководитель просит сделать акцент, одной фразой; пусто = общий обзор")


class Period(BaseModel):
    start: int  # unix, включительно
    end: int  # unix, не включительно

    @property
    def days(self) -> int:
        return round((self.end - self.start) / 86400)

    def previous(self) -> "Period":
        return Period(start=self.start - (self.end - self.start), end=self.start)

    def label(self) -> str:
        fmt = "%d.%m.%Y"
        s = datetime.fromtimestamp(self.start).strftime(fmt)  # локальное время сервера, как и «Сформирован»
        e = datetime.fromtimestamp(self.end - 1).strftime(fmt)
        return f"{s} — {e}"


# ---------- CRM Agent ----------

class ManagerMetrics(BaseModel):
    user_id: int  # 0 = весь отдел
    name: str
    leads_created: int = 0
    leads_created_prev: int = 0
    won_count: int = 0
    won_count_prev: int = 0
    won_amount: int = 0
    won_amount_prev: int = 0
    lost_count: int = 0
    win_rate: float | None = None  # won / (won + lost) за период
    open_leads: int = 0
    leads_without_tasks: int = 0
    leads_overdue: int = 0
    tasks_completed: int = 0
    calls_total: int = 0
    calls_talk: int = 0  # звонки длиннее порога MIN_CALL_SECONDS
    talk_minutes: int = 0
    calls_analyzed: int = 0
    avg_call_score: float | None = None


class ProblemLead(BaseModel):
    lead_id: int
    name: str
    manager_id: int
    manager: str
    price: int
    stage: str
    reason: Literal["no_task", "overdue_task"]
    overdue_days: int = 0
    days_since_update: int = 0


class CrmSnapshot(BaseModel):
    period: Period
    managers: list[ManagerMetrics]
    department: ManagerMetrics
    problem_leads: list[ProblemLead]
    loss_reasons: dict[str, int] = Field(default_factory=dict)  # причина отказа из CRM -> кол-во
    open_by_stage: dict[str, int] = Field(default_factory=dict)
    lead_names: dict[int, str] = Field(default_factory=dict)  # для ссылок в отчёте


# ---------- Call Quality Agent ----------

class CallRecord(BaseModel):
    call_id: int  # id примечания call_in/call_out в amoCRM
    lead_id: int | None
    manager_id: int
    direction: Literal["in", "out"]
    duration: int
    recording_url: str | None
    created_at: int


CallOutcome = Literal["следующий шаг согласован", "отказ", "клиент думает", "нецелевой", "другое"]


class CallReview(BaseModel):
    """То, что LLM возвращает по одному звонку (structured output)."""

    greeting: int = Field(ge=0, le=2, description="Представился, назвал компанию и цель звонка: 0 нет, 1 частично, 2 да")
    needs_discovery: int = Field(ge=0, le=2, description="Выяснил потребность открытыми вопросами")
    presentation: int = Field(ge=0, le=2, description="Презентовал решение под выявленную потребность, а не прайс целиком")
    objection_handling: int = Field(ge=0, le=2, description="Отработал возражения; 2, если возражений не было")
    next_step: int = Field(ge=0, le=2, description="Договорился о конкретном следующем шаге с датой")
    outcome: CallOutcome
    refusal_reason: str | None = Field(None, description="Причина отказа словами клиента, если был отказ")
    mistakes: list[str] = Field(default_factory=list, description="Конкретные ошибки менеджера, до 3 штук")
    summary: str = Field(description="О чём звонок, 1–2 предложения")
    confidence: float = Field(ge=0, le=1, description="Насколько уверенно можно оценить звонок по транскрипту")

    @model_validator(mode="before")
    @classmethod
    def _clamp(cls, data: object) -> object:
        """Constrained decoding не гарантирует minimum/maximum — приводим к шкале, а не роняем звонок."""
        if isinstance(data, dict):
            for k in ("greeting", "needs_discovery", "presentation", "objection_handling", "next_step"):
                if isinstance(data.get(k), (int, float)):
                    data[k] = min(2, max(0, int(data[k])))
            if isinstance(data.get("confidence"), (int, float)):
                data["confidence"] = min(1.0, max(0.0, float(data["confidence"])))
            if isinstance(data.get("mistakes"), list):
                data["mistakes"] = data["mistakes"][:3]
        return data


class CallAnalysis(CallReview):
    call_id: int
    lead_id: int | None
    manager_id: int
    duration: int
    score: int  # 0..100, считается кодом из чек-листа
    method: Literal["llm", "heuristic"]

    @classmethod
    def from_review(cls, call: CallRecord, review: CallReview, method: Literal["llm", "heuristic"]) -> "CallAnalysis":
        points = (
            review.greeting + review.needs_discovery + review.presentation + review.objection_handling + review.next_step
        )
        return cls(
            **review.model_dump(),
            call_id=call.call_id,
            lead_id=call.lead_id,
            manager_id=call.manager_id,
            duration=call.duration,
            score=round(points / 10 * 100),
            method=method,
        )


# ---------- Business Analyst ----------

class Finding(BaseModel):
    title: str = Field(description="Что происходит — коротко")
    explanation: str = Field(description="Почему это происходит и чем грозит, 1–3 предложения")
    severity: Literal["high", "medium", "low"]
    evidence: list[str] = Field(
        description="Ссылки на факты: metric:<user_id|dept>.<поле>, lead:<id>, call:<id>, manager:<id>"
    )


class Recommendation(BaseModel):
    action: str = Field(description="Что сделать, конкретно")
    owner: str = Field(description="Кто делает: руководитель или имя менеджера")
    evidence: list[str] = Field(default_factory=list)


class AnalystOutput(BaseModel):
    headline: str = Field(description="Главное за период в 2–3 предложениях")
    findings: list[Finding]
    recommendations: list[Recommendation]


# ---------- Supervisor: проверка и передача человеку ----------

class HumanReviewItem(BaseModel):
    kind: Literal["call", "finding", "lead", "data"]
    ref: str
    reason: str


class Verification(BaseModel):
    headline: str = ""
    accepted_findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # почему что-то отброшено
    rejected_findings: list[Finding] = Field(default_factory=list)
    invalid_refs: list[str] = Field(default_factory=list)
    human_review: list[HumanReviewItem] = Field(default_factory=list)
