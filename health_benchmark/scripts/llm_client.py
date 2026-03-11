from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .config import BenchmarkConfig
from .validation import GenerationOutput


def _import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("openai is required. Install dependencies in requirements.txt") from exc
    return OpenAI


@dataclass
class LLMCallResult:
    parsed_output: dict[str, Any]
    raw_response: dict[str, Any]
    usage: dict[str, int]
    response_id: str | None
    latency_ms: int


class OpenAILLMClient:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv(self.config.openai.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key in environment variable {self.config.openai.api_key_env}."
                )
            OpenAI = _import_openai()
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self.config.openai.timeout_seconds,
            }
            if self.config.openai.base_url:
                kwargs["base_url"] = self.config.openai.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema: type[BaseModel],
    ) -> LLMCallResult:
        started = time.time()
        response = self.client.responses.parse(
            model=self.config.openai.model,
            instructions=system_message,
            input=user_message,
            temperature=self.config.openai.temperature,
            max_output_tokens=self.config.openai.max_output_tokens,
            text_format=response_schema,
        )
        latency_ms = int((time.time() - started) * 1000)
        parsed_output = response.output_parsed
        if parsed_output is None:
            raise RuntimeError("OpenAI response did not contain parsed structured output.")
        usage = getattr(response, "usage", None)
        return LLMCallResult(
            parsed_output=parsed_output.model_dump(mode="json"),
            raw_response=self._serialize_response(response),
            usage={
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            response_id=getattr(response, "id", None),
            latency_ms=latency_ms,
        )

    def generate_response(self, system_message: str, user_message: str) -> LLMCallResult:
        return self.generate_structured_response(system_message, user_message, GenerationOutput)

    @staticmethod
    def _serialize_response(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump(warnings=False)
            except TypeError:
                return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return {"repr": repr(response)}


def build_llm_client(config: BenchmarkConfig) -> OpenAILLMClient:
    return OpenAILLMClient(config)
