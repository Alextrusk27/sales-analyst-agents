"""Фрагмент для пункта 3 задания гоняется против того же фейкового amoCRM, что и система."""
import httpx
import pytest

from sales_analyst.agents.crm_agent import find_problem_leads
from sales_analyst.demo.fake_amocrm import make_transport
from snippets import overdue_leads

from .conftest import NOW


@pytest.fixture
def patched(monkeypatch):
    def apply(transport):
        real_client = httpx.Client
        monkeypatch.setattr(overdue_leads.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
        monkeypatch.setattr(overdue_leads.time, "sleep", lambda s: None)

    return apply


def test_snippet_matches_crm_agent(dataset, patched):
    patched(make_transport(dataset, fail_first=("/tasks",)))  # заодно проверяем повтор после 429

    found = overdue_leads.problem_leads("demo", "demo-token", now=NOW)

    open_leads = [l for l in dataset.leads if l["status_id"] not in (142, 143)]
    open_tasks = [t for t in dataset.tasks if not t["is_completed"]]
    expected = find_problem_leads(open_leads, open_tasks, NOW)
    assert {l["id"] for l in found["without_tasks"]} == {l["id"] for l, r, _ in expected if r == "no_task"}
    assert {l["id"] for l in found["overdue"]} == {l["id"] for l, r, _ in expected if r == "overdue_task"}
    assert found["without_tasks"] and found["overdue"]
    # есть и сделки с задачей в срок — граница «просрочено / не просрочено» реально проверяется
    assert len(found["without_tasks"]) + len(found["overdue"]) < len(open_leads)


def test_snippet_gives_up_on_endless_429(patched):
    patched(httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(httpx.HTTPStatusError):
        overdue_leads.problem_leads("demo", "t", now=NOW)
