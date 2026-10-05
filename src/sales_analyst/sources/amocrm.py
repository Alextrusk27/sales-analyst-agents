"""Клиент amoCRM REST API v4: только чтение, пагинация, лимиты, ретраи.

Особенности API, которые здесь учтены:
* пустая выборка возвращается как HTTP 204 без тела, а не как пустой список;
* лимит — около 7 запросов в секунду на интеграцию, при превышении 429;
* максимум 250 записей на страницу, следующая страница — в `_links.next`;
* системные статусы 142 («Успешно реализовано») и 143 («Закрыто и не реализовано»)
  одинаковы во всех воронках.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx

log = logging.getLogger(__name__)

WON_STATUS = 142
LOST_STATUS = 143
CLOSED_STATUSES = {WON_STATUS, LOST_STATUS}
PAGE_LIMIT = 250
MAX_STATUS_FILTERS = 60

Params = list[tuple[str, str | int]]


class AmoCrmError(RuntimeError):
    pass


def _retry_after(resp: httpx.Response, default: float) -> float:
    try:
        return min(float(resp.headers.get("Retry-After", default)), 60.0)
    except ValueError:  # Retry-After в формате HTTP-даты
        return default


class AmoCrmClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        max_rps: float = 6.0,
        max_retries: int = 4,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(30.0),
            transport=transport,
        )
        self._min_interval = 1.0 / max_rps
        self._max_retries = max_retries
        self._last_request = 0.0
        self._lock = threading.Lock()

    # ---------- низкий уровень ----------

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get(self, path: str, params: Params | None = None) -> dict[str, Any] | None:
        """GET с ретраями. Возвращает None на 204 (amoCRM так отвечает на пустую выборку)."""
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise AmoCrmError(f"{path}: сеть недоступна: {exc}") from exc
                time.sleep(2**attempt)
                continue
            if resp.status_code == 204:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self._max_retries:
                    raise AmoCrmError(f"{path}: HTTP {resp.status_code} после {attempt + 1} попыток")
                delay = _retry_after(resp, default=2**attempt)
                log.warning("amoCRM %s -> %s, повтор через %.1f с", path, resp.status_code, delay)
                time.sleep(delay)
                continue
            if resp.status_code == 401:
                raise AmoCrmError(f"{path}: HTTP 401 — токен недействителен или истёк")
            if resp.status_code == 403:
                raise AmoCrmError(
                    f"{path}: HTTP 403 — нет прав (например, /users доступен только администратору) "
                    "или IP временно заблокирован за превышение лимита запросов"
                )
            resp.raise_for_status()
            return resp.json()
        raise AmoCrmError(f"{path}: исчерпаны попытки")  # pragma: no cover

    def paginate(self, path: str, key: str, params: Params | None = None) -> Iterator[dict[str, Any]]:
        page = 1
        while True:
            data = self.get(path, [*(params or []), ("limit", PAGE_LIMIT), ("page", page)])
            if data is None:
                return
            yield from data.get("_embedded", {}).get(key, [])
            if "next" not in data.get("_links", {}):
                return
            page += 1

    # ---------- сущности ----------

    def users(self) -> list[dict[str, Any]]:
        return list(self.paginate("/users", "users"))

    def pipelines(self) -> list[dict[str, Any]]:
        data = self.get("/leads/pipelines") or {}
        return data.get("_embedded", {}).get("pipelines", [])

    def leads(self, params: Params) -> list[dict[str, Any]]:
        return list(self.paginate("/leads", "leads", [("with", "loss_reason,contacts"), *params]))

    def leads_created(self, start: int, end: int) -> list[dict[str, Any]]:
        return self.leads([("filter[created_at][from]", start), ("filter[created_at][to]", end - 1)])

    def leads_closed(self, start: int, end: int) -> list[dict[str, Any]]:
        return self.leads([("filter[closed_at][from]", start), ("filter[closed_at][to]", end - 1)])

    def open_leads(self, pipelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Все незакрытые сделки: фильтр по парам (воронка, статус) без 142/143."""
        pairs = [
            (p["id"], s["id"])
            for p in pipelines
            for s in p.get("_embedded", {}).get("statuses", [])
            if s["id"] not in CLOSED_STATUSES
        ]
        if not pairs:
            return []
        if len(pairs) > MAX_STATUS_FILTERS:  # иначе упрёмся в длину URL (414) — фильтруем на клиенте
            return [l for l in self.leads([]) if l["status_id"] not in CLOSED_STATUSES]
        params: Params = []
        for i, (pipeline_id, status_id) in enumerate(pairs):
            params += [(f"filter[statuses][{i}][pipeline_id]", pipeline_id), (f"filter[statuses][{i}][status_id]", status_id)]
        return self.leads(params)

    def open_lead_tasks(self) -> list[dict[str, Any]]:
        return list(
            self.paginate("/tasks", "tasks", [("filter[entity_type]", "leads"), ("filter[is_completed]", 0)])
        )

    def completed_tasks(self, start: int, end: int) -> list[dict[str, Any]]:
        # Отдельного поля «дата выполнения» нет: закрытие задачи меняет updated_at.
        return list(
            self.paginate(
                "/tasks",
                "tasks",
                [
                    ("filter[is_completed]", 1),
                    ("filter[updated_at][from]", start),
                    ("filter[updated_at][to]", end - 1),
                ],
            )
        )

    def call_notes(self, entity_type: str, start: int, end: int) -> list[dict[str, Any]]:
        """Примечания-звонки (call_in/call_out), которые пишет виджет телефонии.

        В params лежат duration, link (ссылка на запись у провайдера), phone, call_status.
        Телефония обычно пишет звонок в контакт, иногда в сделку — поэтому читаем обе сущности.
        """
        return list(
            self.paginate(
                f"/{entity_type}/notes",
                "notes",
                [
                    ("filter[note_type][]", "call_in"),
                    ("filter[note_type][]", "call_out"),
                    ("filter[updated_at][from]", start),
                    ("filter[updated_at][to]", end - 1),
                ],
            )
        )

    def close(self) -> None:
        self._http.close()
