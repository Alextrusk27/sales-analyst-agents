import httpx
import pytest

from sales_analyst.sources.amocrm import AmoCrmClient, AmoCrmError

BASE = "https://t.amocrm.ru/api/v4"


def client(handler) -> AmoCrmClient:
    return AmoCrmClient(BASE, "tok", transport=httpx.MockTransport(handler), max_rps=1000, max_retries=2)


def test_paginates_until_no_next_link():
    seen_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        seen_pages.append(page)
        links = {"next": {"href": "x"}} if page < 3 else {}
        return httpx.Response(200, json={"_links": links, "_embedded": {"leads": [{"id": page}]}})

    assert [l["id"] for l in client(handler).paginate("/leads", "leads")] == [1, 2, 3]
    assert seen_pages == [1, 2, 3]


def test_204_means_empty_result():
    assert client(lambda r: httpx.Response(204)).users() == []


def test_retries_after_429_and_sends_token():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["Authorization"])
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"_embedded": {"users": [{"id": 1}]}})

    assert client(handler).users() == [{"id": 1}]
    assert calls == ["Bearer tok", "Bearer tok"]


def test_gives_up_after_max_retries():
    with pytest.raises(AmoCrmError, match="HTTP 503"):
        client(lambda r: httpx.Response(503, headers={"Retry-After": "0"})).users()


def test_auth_error_is_explicit():
    with pytest.raises(AmoCrmError, match="токен"):
        client(lambda r: httpx.Response(401)).users()


def test_open_leads_filters_out_closed_statuses():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.url.params.multi_items())
        return httpx.Response(204)

    pipelines = [{"id": 7, "_embedded": {"statuses": [{"id": 1}, {"id": 142}, {"id": 143}, {"id": 2}]}}]
    client(handler).open_leads(pipelines)
    assert captured["filter[statuses][0][status_id]"] == "1"
    assert captured["filter[statuses][1][status_id]"] == "2"
    assert "filter[statuses][2][status_id]" not in captured
