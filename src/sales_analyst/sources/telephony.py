"""Записи звонков и распознавание речи.

Метаданные звонков берём из amoCRM (примечания call_in/call_out пишет виджет любой
телефонии: Mango, UIS, Sipuni, Zadarma, Билайн и т.д.), а аудио скачиваем по ссылке
`params.link` у провайдера. Так система не зависит от конкретной АТС: другой провайдер —
другой заголовок авторизации, код тот же.
"""
from __future__ import annotations

import re
from typing import Protocol

import httpx


class RecordingFetcher:
    def __init__(self, auth_header: str = "", transport: httpx.BaseTransport | None = None) -> None:
        headers = {}
        if auth_header and ":" in auth_header:
            name, value = auth_header.split(":", 1)
            headers[name.strip()] = value.strip()
        self._http = httpx.Client(headers=headers, timeout=60.0, follow_redirects=True, transport=transport)

    def fetch(self, url: str) -> bytes:
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp.content


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, filename: str) -> str: ...


class OpenAiCompatibleTranscriber:
    """Любой сервис с OpenAI-совместимым `/v1/audio/transcriptions`.

    Это и облачный Whisper, и self-hosted faster-whisper-server — второй вариант нужен,
    если записи разговоров с клиентами нельзя отдавать во внешний сервис (152-ФЗ).
    """

    def __init__(self, base_url: str, api_key: str, model: str, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=300.0,
            transport=transport,
        )
        self._model = model

    def transcribe(self, audio: bytes, filename: str) -> str:
        resp = self._http.post(
            "/audio/transcriptions",
            files={"file": (filename, audio)},
            data={"model": self._model, "language": "ru", "response_format": "json"},
        )
        resp.raise_for_status()
        return resp.json()["text"]


class DemoTranscriber:
    """Демо: «аудио» в фикстурах — это уже текст разговора в UTF-8."""

    def transcribe(self, audio: bytes, filename: str) -> str:
        return audio.decode("utf-8")


_PHONE = re.compile(r"(?:\+7|\b8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CARD = re.compile(r"\b(?:\d[ -]?){16}\b")


def redact_pii(text: str) -> str:
    """Убираем телефоны, e-mail и номера карт до отправки транскрипта в LLM."""
    text = _CARD.sub("[карта]", text)
    text = _PHONE.sub("[телефон]", text)
    return _EMAIL.sub("[email]", text)
