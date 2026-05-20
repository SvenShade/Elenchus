from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import BadRequestError, OpenAI
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from llm_mcts.config import LLMConfig
from llm_mcts.trace import TraceWriter


class LLMJSONError(ValueError):
    pass


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise LLMJSONError("response did not contain valid JSON")


class LLMClient:
    def __init__(self, config: LLMConfig, tracer: TraceWriter | None = None):
        self.config = config
        self.tracer = tracer
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def close(self) -> None:
        self._client.close()

    @retry(
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(BadRequestError),
        reraise=True,
    )
    def chat(
        self,
        *,
        prompt_name: str,
        messages: list[dict[str, str]],
        state_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        started = time.perf_counter()
        raw_response: str | None = None
        usage: dict[str, Any] | None = None
        error: str | None = None
        try:
            merged_extra_body = dict(self.config.extra_body)
            if extra_body:
                merged_extra_body.update(extra_body)
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
                max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
                extra_body=merged_extra_body or None,
            )
            raw_response = response.choices[0].message.content or ""
            usage_obj = getattr(response, "usage", None)
            usage = usage_obj.model_dump() if usage_obj is not None else None
            return raw_response
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            if self.tracer:
                self.tracer.record_call(
                    prompt_name=prompt_name,
                    messages=messages,
                    raw_response=raw_response,
                    error=error,
                    latency_s=time.perf_counter() - started,
                    state_id=state_id,
                    usage=usage,
                )

    @retry(
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(BadRequestError),
        reraise=True,
    )
    def complete(
        self,
        *,
        prompt_name: str,
        prompt: str,
        state_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        started = time.perf_counter()
        raw_response: str | None = None
        usage: dict[str, Any] | None = None
        error: str | None = None
        trace_messages = [{"role": "completion_prompt", "content": prompt}]
        try:
            merged_extra_body = dict(self.config.extra_body)
            if extra_body:
                merged_extra_body.update(extra_body)
            request_temperature = (
                self.config.turn_temperature
                if temperature is None
                else temperature
            )
            if request_temperature is None:
                request_temperature = self.config.temperature
            request_max_tokens = max_tokens or self.config.turn_max_tokens
            if self.config.turn_transport == "chat":
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=request_temperature,
                    max_tokens=request_max_tokens,
                    stop=stop,
                    extra_body=merged_extra_body or None,
                )
                raw_response = response.choices[0].message.content or ""
            else:
                response = self._client.completions.create(
                    model=self.config.model,
                    prompt=prompt,
                    temperature=request_temperature,
                    max_tokens=request_max_tokens,
                    stop=stop,
                    extra_body=merged_extra_body or None,
                )
                raw_response = response.choices[0].text or ""
            usage_obj = getattr(response, "usage", None)
            usage = usage_obj.model_dump() if usage_obj is not None else None
            return raw_response
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            if self.tracer:
                self.tracer.record_call(
                    prompt_name=prompt_name,
                    messages=trace_messages,
                    raw_response=raw_response,
                    error=error,
                    latency_s=time.perf_counter() - started,
                    state_id=state_id,
                    usage=usage,
                )

    def chat_json(
        self,
        *,
        prompt_name: str,
        messages: list[dict[str, str]],
        state_id: str | None = None,
        fallback: Any | None = None,
        max_repairs: int = 1,
        max_tokens: int | None = None,
    ) -> Any:
        raw = self.chat(
            prompt_name=prompt_name,
            messages=messages,
            state_id=state_id,
            max_tokens=max_tokens,
        )
        try:
            parsed = extract_json_object(raw)
            if self.tracer:
                self.tracer.record_call(
                    prompt_name=f"{prompt_name}.parsed",
                    messages=[],
                    raw_response=raw,
                    parsed=parsed,
                    state_id=state_id,
                )
            return parsed
        except Exception as first_error:
            last_error = first_error

        repair_messages = list(messages)
        for _ in range(max_repairs):
            repair_messages = [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not include markdown or prose.",
                },
                {
                    "role": "user",
                    "content": (
                        "Repair this invalid JSON response so it matches the requested schema.\n\n"
                        f"Original response:\n{raw}"
                    ),
                },
            ]
            repaired = self.chat(
                prompt_name=f"{prompt_name}.repair",
                messages=repair_messages,
                state_id=state_id,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            try:
                parsed = extract_json_object(repaired)
                if self.tracer:
                    self.tracer.record_call(
                        prompt_name=f"{prompt_name}.parsed",
                        messages=[],
                        raw_response=repaired,
                        parsed=parsed,
                        state_id=state_id,
                    )
                return parsed
            except Exception as repair_error:
                last_error = repair_error

        if fallback is not None:
            parsed_fallback = _fallback_with_partial_json_fields(raw, fallback)
            if self.tracer:
                self.tracer.record_call(
                    prompt_name=f"{prompt_name}.fallback",
                    messages=[],
                    raw_response=raw,
                    parsed=parsed_fallback,
                    error=repr(last_error),
                    state_id=state_id,
                )
            return parsed_fallback
        raise LLMJSONError(f"{prompt_name} failed to produce valid JSON: {last_error}") from last_error


def _fallback_with_partial_json_fields(raw: str, fallback: Any) -> Any:
    if not isinstance(fallback, dict):
        return fallback
    patched = dict(fallback)
    for key in ("summary", "next_move_guidance", "rationale"):
        if key not in patched:
            continue
        value = _extract_partial_json_string(raw, key)
        if value:
            patched[key] = value
    return patched


def _extract_partial_json_string(raw: str, key: str) -> str:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"?'
    match = re.search(pattern, raw, flags=re.DOTALL)
    if not match:
        return ""
    text = match.group(1)
    try:
        return json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text.replace('\\"', '"').replace("\\n", "\n").strip()
