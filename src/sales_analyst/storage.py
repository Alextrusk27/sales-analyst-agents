"""Хранилище результатов: SQLite (в проде — та же схема в PostgreSQL).

* transcripts     — кэш распознавания: один звонок распознаётся один раз, повторные отчёты бесплатны;
* call_analyses   — оценка каждого звонка (можно строить динамику по менеджеру);
* runs            — запрос руководителя, план, метрики, проверка и итоговый отчёт.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import CallAnalysis

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   INTEGER NOT NULL,
    request      TEXT NOT NULL,
    plan_json    TEXT,
    metrics_json TEXT,
    verification_json TEXT,
    report_md    TEXT,
    status       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcripts (
    call_id     INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS call_analyses (
    run_id      TEXT NOT NULL,
    call_id     INTEGER NOT NULL,
    manager_id  INTEGER NOT NULL,
    lead_id     INTEGER,
    score       INTEGER NOT NULL,
    method      TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    PRIMARY KEY (run_id, call_id)
);
"""


class Storage:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def start_run(self, run_id: str, request: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO runs(run_id, created_at, request, status) VALUES (?, ?, ?, 'running')",
                (run_id, int(time.time()), request),
            )

    def update_run(self, run_id: str, **fields: str) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._db:
            self._db.execute(f"UPDATE runs SET {cols} WHERE run_id = ?", (*fields.values(), run_id))

    def get_transcript(self, call_id: int) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT text FROM transcripts WHERE call_id = ?", (call_id,)).fetchone()
        return row[0] if row else None

    def save_transcript(self, call_id: int, text: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO transcripts(call_id, text, created_at) VALUES (?, ?, ?)",
                (call_id, text, int(time.time())),
            )

    def save_call_analysis(self, run_id: str, a: CallAnalysis) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO call_analyses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, a.call_id, a.manager_id, a.lead_id, a.score, a.method, a.model_dump_json()),
            )

    def run_report(self, run_id: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT report_md FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def dumps(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)
