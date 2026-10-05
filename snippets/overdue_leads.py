"""amoCRM API v4: открытые сделки без задач или с просроченными задачами."""
import os
import time

import httpx

CLOSED = {142, 143}  # системные статусы «Успешно реализовано» и «Закрыто и не реализовано»


def fetch_all(client: httpx.Client, path: str, key: str, params: dict) -> list[dict]:
    items, page, retries = [], 1, 0
    while True:
        resp = client.get(path, params={**params, "limit": 250, "page": page})
        if resp.status_code == 429 and retries < 5:  # лимит ~7 запросов/с; после 5 попыток — ошибка
            retries += 1
            time.sleep(2**retries)
            continue
        if resp.status_code == 204:  # пустая выборка приходит как 204 без тела
            return items
        resp.raise_for_status()
        data = resp.json()
        items += data["_embedded"][key]
        if "next" not in data.get("_links", {}):
            return items
        page, retries = page + 1, 0


def problem_leads(subdomain: str, token: str, now: float | None = None) -> dict[str, list[dict]]:
    now = time.time() if now is None else now
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=f"https://{subdomain}.amocrm.ru/api/v4", headers=headers, timeout=30) as client:
        pipelines = fetch_all(client, "/leads/pipelines", "pipelines", {})
        open_stages = [(p["id"], s["id"]) for p in pipelines for s in p["_embedded"]["statuses"] if s["id"] not in CLOSED]
        status_filter = {f"filter[statuses][{i}][{k}]": v for i, (pipeline_id, status_id) in enumerate(open_stages)
                         for k, v in (("pipeline_id", pipeline_id), ("status_id", status_id))}
        leads = fetch_all(client, "/leads", "leads", status_filter)  # только сделки на открытых этапах
        tasks = fetch_all(client, "/tasks", "tasks", {"filter[entity_type]": "leads", "filter[is_completed]": 0})
    earliest: dict[int, int] = {}  # самый ранний срок открытой задачи по каждой сделке
    for t in tasks:
        earliest[t["entity_id"]] = min(t["complete_till"], earliest.get(t["entity_id"], t["complete_till"]))
    return {
        "without_tasks": [lead for lead in leads if lead["id"] not in earliest],
        "overdue": [lead for lead in leads if earliest.get(lead["id"], now) < now],
    }


if __name__ == "__main__":
    for kind, leads in problem_leads(os.environ["AMOCRM_SUBDOMAIN"], os.environ["AMOCRM_TOKEN"]).items():
        print(f"{kind}: {len(leads)}", [(lead["id"], lead["name"], lead["responsible_user_id"]) for lead in leads[:20]])
