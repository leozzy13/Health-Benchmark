from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError as PydanticValidationError

from .config import BenchmarkConfig, DEFAULT_VLLM_BASE_URL
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


class BaseLLMClient(Protocol):
    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema: type[BaseModel],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMCallResult: ...

    def generate_response(
        self,
        system_message: str,
        user_message: str,
        *,
        max_output_tokens: int | None = None,
    ) -> LLMCallResult: ...


class OpenAILLMClient:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv(self.config.llm.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key in environment variable {self.config.llm.api_key_env}."
                )
            self._client = _build_openai_client(
                api_key=api_key,
                timeout_seconds=self.config.llm.timeout_seconds,
                base_url=self.config.llm.base_url,
            )
        return self._client

    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema: type[BaseModel],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMCallResult:
        started = time.time()
        response = self.client.responses.parse(
            model=self.config.llm.model,
            instructions=system_message,
            input=user_message,
            temperature=self.config.llm.temperature,
            max_output_tokens=(
                int(max_output_tokens)
                if max_output_tokens is not None
                else self.config.llm.max_output_tokens
            ),
            text_format=response_schema,
        )
        latency_ms = int((time.time() - started) * 1000)
        parsed_output = response.output_parsed
        if parsed_output is None:
            raise RuntimeError("OpenAI response did not contain parsed structured output.")
        usage = getattr(response, "usage", None)
        return LLMCallResult(
            parsed_output=parsed_output.model_dump(mode="json"),
            raw_response=_serialize_response(response),
            usage=_usage_from_response(usage, input_field="input_tokens", output_field="output_tokens"),
            response_id=getattr(response, "id", None),
            latency_ms=latency_ms,
        )

    def generate_response(
        self,
        system_message: str,
        user_message: str,
        *,
        max_output_tokens: int | None = None,
    ) -> LLMCallResult:
        return self.generate_structured_response(
            system_message,
            user_message,
            GenerationOutput,
            max_output_tokens=max_output_tokens,
        )


class VLLMLLMClient:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv(self.config.llm.api_key_env) or "EMPTY"
            self._client = _build_openai_client(
                api_key=api_key,
                timeout_seconds=self.config.llm.timeout_seconds,
                base_url=self.config.llm.base_url or DEFAULT_VLLM_BASE_URL,
            )
        return self._client

    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema: type[BaseModel],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMCallResult:
        return _generate_chat_completion_structured_response(
            client=self.client,
            config=self.config,
            system_message=system_message,
            user_message=user_message,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
            provider_label="vLLM",
            extra_body=self._extra_body(),
        )

    def generate_response(
        self,
        system_message: str,
        user_message: str,
        *,
        max_output_tokens: int | None = None,
    ) -> LLMCallResult:
        return self.generate_structured_response(
            system_message,
            user_message,
            GenerationOutput,
            max_output_tokens=max_output_tokens,
        )

    def _extra_body(self) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": bool(self.config.vllm.enable_thinking),
            }
        }
        if self.config.vllm.top_k is not None:
            extra_body["top_k"] = int(self.config.vllm.top_k)
        return extra_body


def build_llm_client(config: BenchmarkConfig) -> BaseLLMClient:
    provider = str(config.llm.provider).strip().lower()
    if provider == "openai":
        return OpenAILLMClient(config)
    if provider == "vllm":
        return VLLMLLMClient(config)
    raise RuntimeError(f"Unsupported llm provider: {config.llm.provider}")


def _generate_chat_completion_structured_response(
    *,
    client: Any,
    config: BenchmarkConfig,
    system_message: str,
    user_message: str,
    response_schema: type[BaseModel],
    max_output_tokens: int | None,
    provider_label: str,
    extra_body: dict[str, Any] | None,
) -> LLMCallResult:
    started = time.time()
    request_kwargs: dict[str, Any] = {
        "model": config.llm.model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": config.llm.temperature,
        "max_tokens": (
            int(max_output_tokens)
            if max_output_tokens is not None
            else config.llm.max_output_tokens
        ),
    }
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**request_kwargs)
    latency_ms = int((time.time() - started) * 1000)
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError(f"{provider_label} response did not contain any choices.")
    message = getattr(choices[0], "message", None)
    content = _coerce_chat_content(getattr(message, "content", None))
    if not content.strip():
        raise RuntimeError(f"{provider_label} response did not contain JSON content.")
    try:
        parsed_output = response_schema.model_validate_json(content)
    except PydanticValidationError as exc:
        raise RuntimeError(f"{provider_label} response did not match expected schema: {exc}") from exc
    usage = getattr(response, "usage", None)
    return LLMCallResult(
        parsed_output=parsed_output.model_dump(mode="json"),
        raw_response=_serialize_response(response),
        usage=_usage_from_response(usage, input_field="prompt_tokens", output_field="completion_tokens"),
        response_id=getattr(response, "id", None),
        latency_ms=latency_ms,
    )


def _build_openai_client(*, api_key: str, timeout_seconds: int, base_url: str | None) -> Any:
    OpenAI = _import_openai()
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": int(timeout_seconds),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _usage_from_response(usage: Any, *, input_field: str, output_field: str) -> dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, input_field, 0) or 0),
        "output_tokens": int(getattr(usage, output_field, 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _coerce_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for item in content:
            if isinstance(item, str):
                rendered.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    rendered.append(text)
                    continue
                if isinstance(text, dict) and isinstance(text.get("value"), str):
                    rendered.append(text["value"])
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                rendered.append(text)
                continue
            if hasattr(text, "value") and isinstance(text.value, str):
                rendered.append(text.value)
        return "".join(rendered)
    return ""


def _serialize_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(warnings=False)
        except TypeError:
            return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    try:
        return json.loads(json.dumps(response))
    except TypeError:
        return {"repr": repr(response)}
