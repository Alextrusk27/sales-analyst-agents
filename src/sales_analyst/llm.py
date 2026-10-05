"""Тонкая обёртка над LLM: один метод — «системный промпт + данные -> pydantic-объект».

Агенты зависят от протокола, а не от провайдера: заменить Claude на OpenAI/GigaChat/YandexGPT —
это новый класс на 20 строк, графу и тестам всё равно.
"""
from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def invoke(self, schema: type[T], system: str, user: str, *, smart: bool = False) -> T: ...


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        from langchain_anthropic import ChatAnthropic

        common = {
            "api_key": settings.anthropic_api_key,
            "max_tokens": 8000,
            "timeout": 180,
            "max_retries": 3,
        }
        if settings.anthropic_base_url:
            common["base_url"] = settings.anthropic_base_url
        self._fast = ChatAnthropic(model=settings.llm_model_fast, **common)
        self._smart = ChatAnthropic(model=settings.llm_model_smart, **common)

    def invoke(self, schema: type[T], system: str, user: str, *, smart: bool = False) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage

        # Нативные structured outputs (constrained decoding), а не принудительный tool call.
        model = (self._smart if smart else self._fast).with_structured_output(schema, method="json_schema")
        result = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if not isinstance(result, schema):  # pragma: no cover - защита от изменений в langchain
            result = schema.model_validate(result)
        return result


@cache
def prompt(name: str) -> str:
    return resources.files("sales_analyst.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
