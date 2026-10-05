from __future__ import annotations

import pytest

from sales_analyst.config import Settings
from sales_analyst.demo.fake_amocrm import BASE, build_dataset, make_transport
from sales_analyst.graph import Deps
from sales_analyst.sources.amocrm import AmoCrmClient
from sales_analyst.sources.telephony import DemoTranscriber, RecordingFetcher
from sales_analyst.storage import Storage

NOW = 1_790_000_000  # фиксированное «сейчас» для воспроизводимости


@pytest.fixture
def dataset():
    return build_dataset(NOW)


@pytest.fixture
def crm(dataset):
    return AmoCrmClient(BASE, "demo-token", transport=make_transport(dataset, fail_first=()), max_rps=1000)


@pytest.fixture
def make_deps(dataset, tmp_path):
    def factory(llm=None) -> Deps:
        transport = make_transport(dataset)
        return Deps(
            settings=Settings(db_path=":memory:", reports_dir=str(tmp_path / "reports")),
            crm=AmoCrmClient(BASE, "demo-token", transport=transport, max_rps=1000),
            fetcher=RecordingFetcher(transport=transport),
            transcriber=DemoTranscriber(),
            storage=Storage(":memory:"),
            llm=llm,
            now=NOW,
        )

    return factory
