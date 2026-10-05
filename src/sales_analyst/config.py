"""Настройки из переменных окружения (и необязательного файла .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Минимальный загрузчик .env без внешних зависимостей; уже заданные переменные не трогает."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    amocrm_subdomain: str = ""
    amocrm_token: str = ""
    amocrm_domain: str = "amocrm.ru"

    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    llm_model_fast: str = "claude-sonnet-5-5"
    llm_model_smart: str = "claude-opus-5-5"

    stt_base_url: str = ""
    stt_api_key: str = ""
    stt_model: str = "whisper-1"
    recording_auth_header: str = ""

    db_path: str = "data/sales_analyst.sqlite"
    reports_dir: str = "reports"
    max_calls: int = 40
    min_call_seconds: int = 60

    @property
    def amocrm_base_url(self) -> str:
        return f"https://{self.amocrm_subdomain}.{self.amocrm_domain}/api/v4"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        env = os.environ.get
        return cls(
            amocrm_subdomain=env("AMOCRM_SUBDOMAIN", ""),
            amocrm_token=env("AMOCRM_TOKEN", ""),
            amocrm_domain=env("AMOCRM_DOMAIN", "amocrm.ru") or "amocrm.ru",
            anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
            anthropic_base_url=env("ANTHROPIC_BASE_URL", ""),
            llm_model_fast=env("LLM_MODEL_FAST", "") or cls.llm_model_fast,
            llm_model_smart=env("LLM_MODEL_SMART", "") or cls.llm_model_smart,
            stt_base_url=env("STT_BASE_URL", ""),
            stt_api_key=env("STT_API_KEY", ""),
            stt_model=env("STT_MODEL", "") or cls.stt_model,
            recording_auth_header=env("RECORDING_AUTH_HEADER", ""),
            db_path=env("DB_PATH", "") or cls.db_path,
            reports_dir=env("REPORTS_DIR", "") or cls.reports_dir,
            max_calls=int(env("MAX_CALLS", "") or cls.max_calls),
            min_call_seconds=int(env("MIN_CALL_SECONDS", "") or cls.min_call_seconds),
        )
