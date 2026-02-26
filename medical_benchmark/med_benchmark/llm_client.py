from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .config import BenchmarkConfig


def _import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "openai package is required for model calls. Install dependencies in medical_benchmark/requirements.txt"
        ) from exc
    return OpenAI


@dataclass(slots=True)
class LLMAttempt:
    attempt_index: int
    status: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


@dataclass(slots=True)
class LLMCallResult:
    text: str
    raw_response: dict[str, Any]
    attempts: list[LLMAttempt]


class OpenAILLMClient:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv(self.config.model.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key in environment variable {self.config.model.api_key_env}."
                )
            OpenAI = _import_openai()
            self._client = OpenAI(api_key=api_key, timeout=self.config.model.timeout_seconds)
        return self._client

    def generate_with_retries(self, system_message: str, user_message: str) -> LLMCallResult:
        attempts: list[LLMAttempt] = []
        last_error: Exception | None = None
        raw_response: dict[str, Any] = {}
        response_text = ""

        total_attempts = max(1, int(self.config.model.retry_limit))
        for attempt_idx in range(1, total_attempts + 1):
            started = time.time()
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.config.model.name,
                    "temperature": self.config.model.temperature,
                    "max_output_tokens": self.config.model.max_output_tokens,
                    "text": {"format": {"type": "json_object"}},
                    "input": [
                        {"role": "system", "content": [{"type": "input_text", "text": system_message}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_message}]},
                    ],
                }
                if self.config.model.seed is not None:
                    request_kwargs["seed"] = self.config.model.seed
                resp = self.client.responses.create(**request_kwargs)
                latency_ms = int((time.time() - started) * 1000)
                response_text = self._extract_text(resp)
                raw_response = self._serialize_response(resp)
                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage else None
                output_tokens = getattr(usage, "output_tokens", None) if usage else None
                attempts.append(
                    LLMAttempt(
                        attempt_index=attempt_idx,
                        status="ok",
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        error=None,
                    )
                )
                return LLMCallResult(text=response_text, raw_response=raw_response, attempts=attempts)
            except Exception as exc:  # pragma: no cover - network/runtime
                latency_ms = int((time.time() - started) * 1000)
                last_error = exc
                attempts.append(
                    LLMAttempt(
                        attempt_index=attempt_idx,
                        status="error",
                        latency_ms=latency_ms,
                        input_tokens=None,
                        output_tokens=None,
                        error=str(exc),
                    )
                )

        raise RuntimeError(f"Model call failed after {total_attempts} attempts: {last_error}")

    @staticmethod
    def _extract_text(resp: Any) -> str:
        output_text = getattr(resp, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        chunks: list[str] = []
        outputs = getattr(resp, "output", None) or []
        for item in outputs:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
        raise RuntimeError("OpenAI response did not contain text output.")

    @staticmethod
    def _serialize_response(resp: Any) -> dict[str, Any]:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
        return {"repr": repr(resp)}
