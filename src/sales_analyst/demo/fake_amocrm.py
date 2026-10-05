"""Демо-режим: фейковый amoCRM API v4 и фейковая телефония на httpx.MockTransport.

Важно: в демо работает настоящий AmoCrmClient — с пагинацией, фильтрами, ответами 204
и повтором после 429. Подменяется только сеть. Данные генерируются детерминированно
относительно «сейчас», чтобы окно «последние 30 дней» всегда было заполнено.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl

import httpx

from .transcripts import TRANSCRIPTS

DAY = 86400
BASE = "https://demo.amocrm.ru/api/v4"
RECORDINGS_HOST = "https://records.demo-telephony.local"

USERS = [
    {"id": 1000, "name": "Администратор", "email": "admin@paypoint.demo"},
    {"id": 1001, "name": "Анна Смирнова", "email": "anna@paypoint.demo"},
    {"id": 1002, "name": "Игорь Петров", "email": "igor@paypoint.demo"},
    {"id": 1003, "name": "Мария Козлова", "email": "maria@paypoint.demo"},
    {"id": 1004, "name": "Дмитрий Орлов", "email": "dmitry@paypoint.demo"},
]
STAGES = [(101, "Новая заявка"), (102, "Квалификация"), (103, "Отправлено КП"), (104, "Переговоры"), (105, "Счёт / договор")]
PIPELINES = [{
    "id": 5001,
    "name": "Продажи",
    "_embedded": {"statuses": [
        *({"id": sid, "name": name} for sid, name in STAGES),
        {"id": 142, "name": "Успешно реализовано"},
        {"id": 143, "name": "Закрыто и не реализовано"},
    ]},
}]
LOSS_REASONS = {1: "Дорого", 2: "Выбрали конкурента", 3: "Не вышли на связь", 4: "Неактуально"}

# Профили менеджеров: сколько сделок за 90 дней, вероятности выигрыша/проигрыша,
# доля открытых сделок без задачи и с просроченной задачей, веса причин отказа.
PROFILES = {
    1001: dict(leads=62, win=0.35, lost=0.22, no_task=0.04, overdue=0.10, tasks_done=95, reasons=[1, 2, 2, 4]),
    1002: dict(leads=64, win=0.14, lost=0.34, no_task=0.25, overdue=0.32, tasks_done=31, reasons=[3, 3, 2, 4]),
    1003: dict(leads=38, win=0.40, lost=0.15, no_task=0.05, overdue=0.05, tasks_done=70, reasons=[1, 4]),
    1004: dict(leads=46, win=0.10, lost=0.40, no_task=0.15, overdue=0.18, tasks_done=40, reasons=[1, 1, 1, 2]),
}
CLIENTS = [
    "Кофейня «Зерно»", "ИП Сидоров", "Цветы «Лаванда»", "Аптека «Здоровье»", "Пекарня «Каравай»",
    "Барбершоп «Бритва»", "Магазин «Уют»", "СТО «Мотор»", "Студия йоги «Прана»", "Кафе «Бульвар»",
    "Зоомагазин «Хвост»", "Химчистка «Лотос»", "Шаурма «Восток»", "Салон «Бьюти»", "Автомойка «Блеск»",
]
PRODUCTS = ["эквайринг", "онлайн-касса", "интернет-эквайринг", "смарт-терминал", "кассы на сеть"]

# Звонки с записью: (менеджер, ключ транскрипта, длительность, направление, дней назад)
TALKS = [
    (1001, "anna_1", 245, "call_out", 3), (1001, "anna_2", 180, "call_out", 9), (1001, "anna_3", 160, "call_in", 15),
    (1002, "igor_1", 75, "call_out", 4), (1002, "igor_2", 62, "call_out", 11), (1002, "igor_3", 64, "call_out", 20),
    (1003, "maria_1", 310, "call_out", 2), (1003, "maria_2", 205, "call_out", 8), (1003, "maria_3", 70, "call_in", 13),
    (1004, "dmitry_1", 95, "call_out", 5), (1004, "dmitry_2", 66, "call_out", 12), (1004, "dmitry_3", 71, "call_out", 18),
]
SHORT_CALLS = {1001: 34, 1002: 12, 1003: 21, 1004: 27}  # недозвоны/автоответчики за период


@dataclass
class Dataset:
    users: list[dict[str, Any]]
    pipelines: list[dict[str, Any]]
    leads: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"leads": [], "contacts": []})
    recordings: dict[str, bytes] = field(default_factory=dict)


def build_dataset(now: int, seed: int = 7) -> Dataset:
    rng = random.Random(seed)
    ds = Dataset(users=USERS, pipelines=PIPELINES)
    lead_id, task_id, note_id = 30_000_000, 50_000_000, 70_000_000

    for uid, p in PROFILES.items():
        for _ in range(p["leads"]):
            lead_id += 1
            created = now - rng.randint(0, 89) * DAY - rng.randint(0, DAY - 1)
            contact_id = lead_id + 10_000_000
            lead: dict[str, Any] = {
                "id": lead_id,
                "name": f"{rng.choice(CLIENTS)} — {rng.choice(PRODUCTS)}",
                "price": rng.randrange(15_000, 450_000, 1_000),
                "responsible_user_id": uid,
                "pipeline_id": 5001,
                "created_at": created,
                "updated_at": created,
                "closed_at": None,
                "closest_task_at": None,
                "loss_reason_id": None,
                "_embedded": {"contacts": [{"id": contact_id, "is_main": True}], "loss_reason": []},
            }
            roll = rng.random()
            age = (now - created) // DAY
            if roll < p["win"] + p["lost"] and age >= 3:
                closed = created + rng.randint(2, max(2, int(age))) * DAY // 2
                lead["closed_at"] = lead["updated_at"] = min(closed, now - 3600)
                if roll < p["win"]:
                    lead["status_id"] = 142
                else:
                    lead["status_id"] = 143
                    reason = rng.choice(p["reasons"])
                    lead["loss_reason_id"] = reason
                    lead["_embedded"]["loss_reason"] = [{"id": reason, "name": LOSS_REASONS[reason]}]
            else:
                lead["status_id"] = rng.choice(STAGES)[0]
                lead["updated_at"] = min(now - 3600, created + rng.randint(0, 20) * DAY)
                hygiene = rng.random()
                if hygiene < p["no_task"]:
                    pass  # нет открытой задачи
                else:
                    task_id += 1
                    overdue = hygiene < p["no_task"] + p["overdue"]
                    deadline = now - rng.randint(1, 25) * DAY if overdue else now + rng.randint(1, 10) * DAY
                    ds.tasks.append({
                        "id": task_id, "entity_id": lead_id, "entity_type": "leads", "responsible_user_id": uid,
                        "is_completed": False, "complete_till": deadline, "text": "Связаться с клиентом",
                        "created_at": created, "updated_at": created,
                    })
                    lead["closest_task_at"] = deadline
            ds.leads.append(lead)

        for _ in range(p["tasks_done"]):
            task_id += 1
            done = now - rng.randint(0, 29) * DAY - rng.randint(0, DAY - 1)
            ds.tasks.append({
                "id": task_id, "entity_id": rng.choice([l["id"] for l in ds.leads if l["responsible_user_id"] == uid]),
                "entity_type": "leads", "responsible_user_id": uid, "is_completed": True,
                "complete_till": done, "text": "Перезвонить", "created_at": done - DAY, "updated_at": done,
            })

    open_by_manager = {
        uid: [l for l in ds.leads if l["responsible_user_id"] == uid and l["status_id"] not in (142, 143)]
        for uid in PROFILES
    }

    def add_call(uid: int, duration: int, note_type: str, days_ago: int, text: str | None, unlinked: bool = False) -> None:
        nonlocal note_id
        note_id += 1
        lead = rng.choice(open_by_manager[uid])
        on_lead = rng.random() < 0.3
        entity = "leads" if on_lead else "contacts"
        entity_id = lead["id"] if on_lead else lead["_embedded"]["contacts"][0]["id"]
        if unlinked:
            entity, entity_id = "contacts", 99_000_000 + note_id
        ts = now - days_ago * DAY - rng.randint(0, DAY // 2)
        link = f"{RECORDINGS_HOST}/rec/{note_id}.mp3"
        if text is not None:
            ds.recordings[link] = text.encode("utf-8")
        ds.notes[entity].append({
            "id": note_id, "entity_id": entity_id, "note_type": note_type, "responsible_user_id": uid,
            "created_by": 0, "created_at": ts, "updated_at": ts,
            "params": {"uniq": f"demo-{note_id}", "duration": duration, "source": "demo-pbx",
                       "link": link, "phone": "+7 999 000-00-00", "call_status": 4},
        })

    for uid, key, duration, note_type, days_ago in TALKS:
        add_call(uid, duration, note_type, days_ago, TRANSCRIPTS[key])
    for uid, n in SHORT_CALLS.items():
        for i in range(n):
            add_call(uid, rng.randint(3, 40), "call_out", rng.randint(0, 29), None, unlinked=(i % 9 == 0))
    return ds


# ---------- фейковый HTTP ----------

def _page(items: list[dict[str, Any]], key: str, query: dict[str, list[str]], path: str) -> httpx.Response:
    limit = int(query.get("limit", ["50"])[0])
    page = int(query.get("page", ["1"])[0])
    chunk = items[(page - 1) * limit: page * limit]
    if not chunk:
        return httpx.Response(204)
    links = {"self": {"href": f"{BASE}{path}?page={page}"}}
    if page * limit < len(items):
        links["next"] = {"href": f"{BASE}{path}?page={page + 1}"}
    return httpx.Response(200, json={"_page": page, "_links": links, "_embedded": {key: chunk}})


def _range(query: dict[str, list[str]], name: str) -> tuple[int, int] | None:
    lo, hi = query.get(f"filter[{name}][from]"), query.get(f"filter[{name}][to]")
    if lo is None and hi is None:
        return None
    return int((lo or ["0"])[0]), int((hi or [str(2**62)])[0])


def _in_range(value: int | None, rng: tuple[int, int] | None) -> bool:
    return rng is None or (value is not None and rng[0] <= value <= rng[1])


def make_transport(ds: Dataset, fail_first: tuple[str, ...] = ("/leads/notes",)) -> httpx.MockTransport:
    """Транспорт, отвечающий как amoCRM. fail_first — пути, где первый запрос получит 429."""
    pending_429 = set(fail_first)

    def handler(request: httpx.Request) -> httpx.Response:
        url = request.url
        if f"{url.scheme}://{url.host}" == RECORDINGS_HOST:
            body = ds.recordings.get(str(url))
            return httpx.Response(200, content=body) if body is not None else httpx.Response(404)
        if request.headers.get("Authorization") != "Bearer demo-token":
            return httpx.Response(401, json={"title": "Unauthorized"})

        path = url.path.removeprefix("/api/v4")
        query: dict[str, list[str]] = {}
        for k, v in parse_qsl(url.query.decode(), keep_blank_values=True):
            query.setdefault(k, []).append(v)

        if path in pending_429:
            pending_429.discard(path)
            return httpx.Response(429, headers={"Retry-After": "0.1"})

        routes: dict[str, Callable[[], httpx.Response]] = {
            "/users": lambda: _page(ds.users, "users", query, path),
            "/leads/pipelines": lambda: httpx.Response(200, json={"_embedded": {"pipelines": ds.pipelines}}),
            "/leads": lambda: _page(_filter_leads(ds, query), "leads", query, path),
            "/tasks": lambda: _page(_filter_tasks(ds, query), "tasks", query, path),
            "/leads/notes": lambda: _page(_filter_notes(ds.notes["leads"], query), "notes", query, path),
            "/contacts/notes": lambda: _page(_filter_notes(ds.notes["contacts"], query), "notes", query, path),
        }
        route = routes.get(path)
        return route() if route else httpx.Response(404, content=json.dumps({"title": "Not found"}))

    return httpx.MockTransport(handler)


def _filter_leads(ds: Dataset, q: dict[str, list[str]]) -> list[dict[str, Any]]:
    created, closed = _range(q, "created_at"), _range(q, "closed_at")
    statuses = {
        (int(q[f"filter[statuses][{i}][pipeline_id]"][0]), int(q[f"filter[statuses][{i}][status_id]"][0]))
        for i in range(1000) if f"filter[statuses][{i}][status_id]" in q
    }
    with_ = (q.get("with") or [""])[0].split(",")
    out = []
    for l in ds.leads:
        if not _in_range(l["created_at"], created) or not _in_range(l["closed_at"], closed):
            continue
        if statuses and (l["pipeline_id"], l["status_id"]) not in statuses:
            continue
        item = {k: v for k, v in l.items() if k != "_embedded"}
        item["_embedded"] = {k: v for k, v in l["_embedded"].items() if k in with_}
        out.append(item)
    return out


def _filter_tasks(ds: Dataset, q: dict[str, list[str]]) -> list[dict[str, Any]]:
    updated = _range(q, "updated_at")
    entity = (q.get("filter[entity_type]") or [None])[0]
    done = (q.get("filter[is_completed]") or [None])[0]
    return [
        t for t in ds.tasks
        if (entity is None or t["entity_type"] == entity)
        and (done is None or t["is_completed"] == (done in ("1", "true")))
        and _in_range(t["updated_at"], updated)
    ]


def _filter_notes(notes: list[dict[str, Any]], q: dict[str, list[str]]) -> list[dict[str, Any]]:
    types = set(q.get("filter[note_type][]", []))
    updated = _range(q, "updated_at")
    return [n for n in notes if (not types or n["note_type"] in types) and _in_range(n["updated_at"], updated)]
