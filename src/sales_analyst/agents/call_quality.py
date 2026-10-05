"""AI Call Quality Agent: звонки из amoCRM -> запись -> текст -> оценка по скрипту.

Конвейер на один звонок: скачать запись -> распознать (с кэшем) -> вычистить ПДн ->
оценить по чек-листу (LLM, structured output) -> посчитать балл кодом.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from ..llm import StructuredLLM, prompt
from ..models import CallAnalysis, CallRecord, CallReview, Period
from ..sources.amocrm import AmoCrmClient
from ..sources.telephony import RecordingFetcher, Transcriber, redact_pii
from ..storage import Storage

log = logging.getLogger(__name__)


RawNote = tuple[str, dict[str, Any]]


def fetch_call_notes(client: AmoCrmClient, period: Period) -> list[RawNote]:
    """Шаг 1 (параллельно с CRM Agent): все примечания-звонки за период по сделкам и контактам."""
    return [
        *(("leads", n) for n in client.call_notes("leads", period.start, period.end)),
        *(("contacts", n) for n in client.call_notes("contacts", period.start, period.end)),
    ]


def build_call_records(
    notes: list[RawNote], contact_to_lead: dict[int, int], manager_ids: set[int], period: Period | None = None
) -> list[CallRecord]:
    """Шаг 2 (после CRM Agent): привязка звонков к сделкам через контакты и фильтр по менеджерам."""
    calls: dict[str, CallRecord] = {}
    for entity, n in notes:
        params = n.get("params") or {}
        # call_responsible — кто реально говорил; примечание может быть создано от имени интеграции.
        responsible = params.get("call_responsible")
        manager_id = responsible if isinstance(responsible, int) else n.get("responsible_user_id") or n.get("created_by") or 0
        if manager_id not in manager_ids:
            continue
        # Выборка API — по updated_at; старый звонок с отредактированным примечанием отсекаем по created_at.
        if period is not None and not period.start <= n["created_at"] < period.end:
            continue
        lead_id = n["entity_id"] if entity == "leads" else contact_to_lead.get(n["entity_id"])
        record = CallRecord(
            call_id=n["id"],
            lead_id=lead_id,
            manager_id=manager_id,
            direction="in" if n["note_type"] == "call_in" else "out",
            duration=int(params.get("duration") or 0),
            recording_url=params.get("link") or None,
            created_at=n["created_at"],
        )
        # Виджет иногда пишет один звонок и в контакт, и в сделку: склеиваем по uniq.
        key = str(params.get("uniq") or n["id"])
        if key not in calls or (calls[key].lead_id is None and record.lead_id is not None):
            calls[key] = record
    return sorted(calls.values(), key=lambda c: c.created_at)


def select_for_analysis(
    calls: list[CallRecord], min_seconds: int, max_calls: int, only_managers: set[int] | None = None
) -> list[CallRecord]:
    """Выборка для STT+LLM: только разговоры с записью, поровну на менеджера, свежие первыми.

    Слушать всё подряд дорого: недозвоны и автоответчики отсекаем по длительности,
    а лимит делим между менеджерами, чтобы сравнение было честным.
    """
    queues: dict[int, list[CallRecord]] = defaultdict(list)
    for c in sorted(calls, key=lambda c: -c.created_at):
        if c.duration >= min_seconds and c.recording_url and (only_managers is None or c.manager_id in only_managers):
            queues[c.manager_id].append(c)
    selected: list[CallRecord] = []
    while len(selected) < max_calls and any(queues.values()):
        for uid in sorted(queues):
            if queues[uid] and len(selected) < max_calls:
                selected.append(queues[uid].pop(0))
    return selected


def transcribe_call(call: CallRecord, fetcher: RecordingFetcher, transcriber: Transcriber, storage: Storage) -> str:
    cached = storage.get_transcript(call.call_id)
    if cached is not None:
        return cached
    audio = fetcher.fetch(call.recording_url or "")
    text = transcriber.transcribe(audio, filename=f"call-{call.call_id}.mp3")
    storage.save_transcript(call.call_id, text)
    return text


def review_with_llm(transcript: str, call: CallRecord, llm: StructuredLLM) -> CallReview:
    user = (
        f"Направление: {'входящий' if call.direction == 'in' else 'исходящий'}, длительность {call.duration} с.\n"
        f"<transcript>\n{redact_pii(transcript)}\n</transcript>"
    )
    return llm.invoke(CallReview, prompt("call_review"), user)


# ---------- офлайн-эвристика (нет ключа LLM) ----------

_OBJECTIONS = ("дорого", "подумаю", "подумать", "не актуально", "неактуально", "у нас уже", "не нужно", "не интересно")
_REFUSAL = ("не интересно", "не актуально", "неактуально", "откажемся", "не нужно", "не будем")
_HANDLING = ("понимаю", "давайте", "если сравнить", "именно поэтому", "могу предложить")
_NEXT_STEP = ("договорились", "созвонимся", "встреч", "отправлю", "пришлю", "выставлю счёт", "выставлю счет")
_WHEN = re.compile(r"завтра|понедельник|вторник|сред[ау]|четверг|пятниц|\b\d{1,2}[:.]\d{2}\b|\bв \d{1,2}\b")


def review_heuristic(transcript: str) -> CallReview:
    t = transcript.lower()
    greeting = 2 if "меня зовут" in t and "компани" in t else 1 if ("здравствуйте" in t or "добрый" in t) else 0
    questions = t.count("?")
    needs = 2 if questions >= 4 else 1 if questions >= 2 else 0
    presentation = 2 if any(w in t for w in ("подойд", "под вашу", "для вашей")) else 1 if "предлага" in t else 0
    objection = any(w in t for w in _OBJECTIONS)
    handled = objection and any(w in t for w in _HANDLING)
    objection_handling = 2 if not objection or handled else 0
    step_words = any(w in t for w in _NEXT_STEP)
    next_step = 2 if step_words and _WHEN.search(t) else 1 if step_words else 0
    refused = any(w in t for w in _REFUSAL)
    outcome = (
        "отказ" if refused and next_step < 2
        else "следующий шаг согласован" if next_step == 2
        else "клиент думает" if "подума" in t
        else "другое"
    )
    mistakes = []
    if needs == 0:
        mistakes.append("не выяснил потребность")
    if objection and not handled:
        mistakes.append("не отработал возражение")
    if next_step == 0:
        mistakes.append("не договорился о следующем шаге")
    return CallReview(
        greeting=greeting,
        needs_discovery=needs,
        presentation=presentation,
        objection_handling=objection_handling,
        next_step=next_step,
        outcome=outcome,  # type: ignore[arg-type]
        refusal_reason=None,
        mistakes=mistakes,
        summary="Оценка по ключевым словам (LLM не подключена).",
        confidence=0.5,
    )


def analyze_call(
    call: CallRecord,
    fetcher: RecordingFetcher,
    transcriber: Transcriber,
    storage: Storage,
    llm: StructuredLLM | None,
) -> tuple[CallAnalysis, str | None]:
    """Возвращает оценку и предупреждение, если пришлось откатиться на эвристику."""
    transcript = transcribe_call(call, fetcher, transcriber, storage)
    if llm is not None:
        try:
            return CallAnalysis.from_review(call, review_with_llm(transcript, call, llm), "llm"), None
        except Exception as exc:  # LLM недоступна или вернула мусор — звонок не теряем
            log.warning("call %s: LLM %s, оцениваю эвристикой", call.call_id, exc)
            warning = f"часть звонков оценена эвристикой: LLM не ответила ({type(exc).__name__})"
            return CallAnalysis.from_review(call, review_heuristic(transcript), "heuristic"), warning
    return CallAnalysis.from_review(call, review_heuristic(transcript), "heuristic"), None
