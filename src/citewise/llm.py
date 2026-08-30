"""Anthropic wrapper producing validated pydantic objects.

The API is asked for a JSON object matching the target schema via
`output_config.format`. That makes malformed output rare but not impossible —
truncation at `max_tokens` and refusals both bypass the constraint — so the
wrapper still parses and validates itself, and on failure retries **once** with
the error fed back to the model before giving up.

The client is constructor-injected so tests can swap in a fake. `search.py`
follows the same convention.
"""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from citewise.config import Config

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 2  # one initial call plus one repair

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """The model did not return output matching the requested schema."""


def to_strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Render a pydantic model as a JSON schema the API will accept.

    Pydantic omits `additionalProperties`, which structured outputs require to
    be `false` on every object. Walks `$defs` as well as the root so nested
    models are covered.
    """

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for item in node:
                tighten(item)

    rendered = schema.model_json_schema()
    tighten(rendered)
    return rendered


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a response that may be fenced or chatty.

    Cheap to try and it saves a repair round-trip, which is a real API call.
    """
    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _response_text(message: Any) -> str:
    return "".join(
        block.text for block in (message.content or []) if getattr(block, "type", None) == "text"
    )


def _repair_prompt(raw: str, problem: str) -> str:
    return (
        f"Your previous response {problem}\n\n"
        f"Error:\n{raw}\n\n"
        "Return only the corrected JSON object. No prose, no code fence, no "
        "explanation — the response must begin with { and end with }."
    )


class StructuredLLM:
    """Injectable Anthropic wrapper.

    `client` is anything exposing `messages.create(...)`; production passes an
    `anthropic.Anthropic`, tests pass `FakeAnthropic`.
    """

    def __init__(self, client: Any = None, config: Config | None = None):
        self.config = config or Config()
        self.client = client if client is not None else self._build_client()
        # Per-call token usage, so the eval can price a run. Repair attempts are
        # recorded too — a retry is real spend and hiding it would understate cost.
        self.usage: list[dict[str, Any]] = []

    def reset_usage(self) -> None:
        self.usage = []

    def usage_totals(self) -> dict[str, int]:
        return {
            "calls": len(self.usage),
            "input_tokens": sum(u["input_tokens"] for u in self.usage),
            "output_tokens": sum(u["output_tokens"] for u in self.usage),
        }

    def _record_usage(self, model: str, message: Any) -> None:
        usage = getattr(message, "usage", None)
        self.usage.append(
            {
                "model": model,
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
        )

    def _build_client(self) -> Any:
        import anthropic

        self.config.require_keys()
        return anthropic.Anthropic(api_key=self.config.anthropic_api_key)

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        model: str,
        max_tokens: int | None = None,
    ) -> T:
        """Return a validated instance of `schema`, repairing once if needed."""
        json_schema = to_strict_json_schema(schema)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        failures: list[str] = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens or self.config.max_tokens,
                system=system,
                messages=messages,
                output_config={"format": {"type": "json_schema", "schema": json_schema}},
            )
            self._record_usage(model, message)

            if getattr(message, "stop_reason", None) == "refusal":
                # Not a formatting problem — a retry would only refuse again.
                raise StructuredOutputError(f"{model} refused the request for {schema.__name__}")

            raw = _response_text(message)
            parsed, problem, detail = self._parse(raw, schema)
            if problem is None:
                return parsed

            failures.append(f"attempt {attempt}: {problem} ({detail[:400]})")
            if attempt == MAX_ATTEMPTS:
                break

            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _repair_prompt(detail, problem)},
            ]

        raise StructuredOutputError(
            f"{model} failed to produce a valid {schema.__name__} after "
            f"{MAX_ATTEMPTS} attempts: " + "; ".join(failures)
        )

    @staticmethod
    def _parse(raw: str, schema: type[T]) -> tuple[T | None, str | None, str]:
        """Returns (instance, problem, detail); `problem` is None on success."""
        if not raw.strip():
            return None, "was empty", "the response contained no text content"

        try:
            payload = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            return None, "did not parse as JSON", str(exc)

        try:
            return schema.model_validate(payload), None, ""
        except ValidationError as exc:
            return None, f"did not validate against the {schema.__name__} schema", str(exc)
