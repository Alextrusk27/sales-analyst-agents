"""Точка входа: `sales-analyst "Проанализируй работу отдела продаж за последние 30 дней" [--demo] [--offline]`."""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import Settings
from .graph import Deps, run
from .llm import AnthropicLLM
from .sources.amocrm import AmoCrmClient
from .sources.telephony import DemoTranscriber, OpenAiCompatibleTranscriber, RecordingFetcher
from .storage import Storage

DEFAULT_REQUEST = "Проанализируй работу отдела продаж за последние 30 дней"


def build_deps(settings: Settings, *, demo: bool, offline: bool) -> Deps:
    llm = None if offline or not settings.llm_enabled else AnthropicLLM(settings)
    storage = Storage(settings.db_path)

    if demo:
        from .demo.fake_amocrm import BASE, build_dataset, make_transport

        now = int(time.time())
        transport = make_transport(build_dataset(now))
        return Deps(
            settings=settings,
            crm=AmoCrmClient(BASE, "demo-token", transport=transport, max_rps=1000),
            fetcher=RecordingFetcher(transport=transport),
            transcriber=DemoTranscriber(),
            storage=storage,
            llm=llm,
            now=now,
        )

    missing = [n for n, v in (("AMOCRM_SUBDOMAIN", settings.amocrm_subdomain), ("AMOCRM_TOKEN", settings.amocrm_token)) if not v]
    if missing:
        sys.exit(f"Не заданы {', '.join(missing)}. Заполните .env или запустите с --demo.")
    return Deps(
        settings=settings,
        crm=AmoCrmClient(settings.amocrm_base_url, settings.amocrm_token),
        fetcher=RecordingFetcher(settings.recording_auth_header),
        # Без STT система работает: звонки считаются по CRM, но не прослушиваются (это попадёт в отчёт).
        transcriber=(
            OpenAiCompatibleTranscriber(settings.stt_base_url, settings.stt_api_key, settings.stt_model)
            if settings.stt_base_url
            else None
        ),
        storage=storage,
        llm=llm,
        crm_url=f"https://{settings.amocrm_subdomain}.{settings.amocrm_domain}",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Мультиагентный анализ отдела продаж (amoCRM + звонки)")
    parser.add_argument("request", nargs="?", default=DEFAULT_REQUEST, help="Запрос руководителя")
    parser.add_argument("--demo", action="store_true", help="Демо-данные вместо реального amoCRM")
    parser.add_argument("--offline", action="store_true", help="Без LLM: правила и эвристики")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = Settings.from_env()
    deps = build_deps(settings, demo=args.demo, offline=args.offline)
    if deps.llm is None and not args.offline:
        print("ANTHROPIC_API_KEY не задан — работаю в офлайн-режиме (правила вместо LLM).", file=sys.stderr)
    state = run(args.request, deps)
    print(state["report"])
    print(f"\nОтчёт сохранён: {state['report_path']}", file=sys.stderr)


if __name__ == "__main__":
    main()
