"""An answerer that uses any model on OpenRouter, through its OpenAI-compatible API.

It sends the same passages, system prompt and answer schema as
:class:`~policy_assistant.answerers.ClaudeAnswerer`, so the agent checks its citations the
same way. It needs an OpenRouter API key in ``OPENROUTER_API_KEY``. The model is
configurable: pass ``model=``, set ``OPENROUTER_MODEL``, or use ``--answerer
openrouter:<model id>`` on the command line. The default is a free model, so trying it
costs nothing, but OpenRouter limits free models to a small number of requests a day.

Structured outputs are asked for first (``response_format`` with a strict JSON schema,
routed only to providers that support it). If the model has no such provider, the
answerer falls back once to asking for the JSON in the prompt and parsing it from the
reply, and keeps doing that for the rest of the run.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from .agent import Draft
from .answerers import SYSTEM_PROMPT, answer_schema, format_passages, parse_answer
from .index import Passage

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
APP_TITLE = "grounded-policy-assistant"

# HTTP statuses worth waiting out: rate limits and provider trouble.
_RETRY = {408, 429, 500, 502, 503, 504}
# Statuses that mean the structured request itself was not accepted for this model.
_NO_STRUCTURED = {400, 404, 422}

JSON_INSTRUCTIONS = """\

Reply with one JSON object and nothing else, no prose and no code fence. It must match \
this JSON schema:
{schema}"""


class OpenRouterError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"OpenRouter returned {status}: {message}")
        self.status = status


class RateLimited(OpenRouterError):
    """OpenRouter kept answering 429 (free models allow only a few requests a day).

    ``daily`` is True when the reply says the day's quota is used up, and ``reset`` is when
    OpenRouter says the limit resets (seconds since the epoch), if it says.
    """

    def __init__(self, status: int, message: str, daily: bool = False, reset=None):
        if reset is not None:
            when = datetime.fromtimestamp(reset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            message += f" (resets {when})"
        super().__init__(status, message)
        self.daily = daily
        self.reset = reset


class OpenRouterAnswerer:
    """Answers from the passages with a model on OpenRouter.

    ``structured`` is ``"auto"`` (try structured outputs, fall back to JSON in the
    prompt), ``True`` (structured outputs only) or ``False`` (JSON in the prompt only).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        structured: bool | str = "auto",
        max_tokens: int = 4000,
        retries: int = 3,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("set OPENROUTER_API_KEY to use the OpenRouter answerer")
        if structured not in ("auto", True, False):
            raise ValueError("structured must be 'auto', True or False")
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout)
        self.client = client
        self.model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        self.name = f"openrouter:{self.model}"
        self.structured = structured
        # How the last draft was asked for: "structured" (response_format) or "prompt"
        # (JSON asked for in the system prompt). None before the first draft.
        self.last_mode: str | None = None
        # How many requests this answerer has sent, retries included.
        self.requests = 0
        self.max_tokens = max_tokens
        self.retries = retries
        self.sleep = sleep
        self._headers = {"Authorization": f"Bearer {api_key}", "X-OpenRouter-Title": APP_TITLE}

    def request(
        self, question: str, passages: Sequence[Passage], feedback: str = "", structured=True
    ) -> dict:
        """The JSON body sent to the chat completions endpoint."""
        schema = answer_schema(len(passages))
        system = SYSTEM_PROMPT
        if not structured:
            system += JSON_INSTRUCTIONS.format(schema=json.dumps(schema))
        prompt = f"{format_passages(passages)}\n\nQuestion: {question}"
        if feedback:
            prompt += f"\n\n{feedback}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
            body["provider"] = {"require_parameters": True}
        return body

    def draft(self, question: str, passages: Sequence[Passage], feedback: str = "") -> Draft:
        structured = self.structured is not False
        try:
            data = self._post(self.request(question, passages, feedback, structured))
        except OpenRouterError as err:
            if not (structured and self.structured == "auto" and err.status in _NO_STRUCTURED):
                raise
            self.structured = False  # this model has no structured-output provider
            structured = False
            data = self._post(self.request(question, passages, feedback, structured=False))
        self.last_mode = "structured" if structured else "prompt"
        return self._read(data, passages)

    def _post(self, body: dict) -> dict:
        for attempt in range(self.retries + 1):
            resp = self.client.post(OPENROUTER_URL, json=body, headers=self._headers)
            self.requests += 1
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:  # some failures arrive with status 200
                    err = data["error"] or {}
                    raise OpenRouterError(int(err.get("code") or 502), _message(data))
                return data
            message = _message(_json(resp))
            daily = resp.status_code == 429 and "per-day" in message
            if resp.status_code in _RETRY and attempt < self.retries and not daily:
                self.sleep(_wait(resp, attempt))
                continue
            if resp.status_code == 429:
                raise RateLimited(429, message, daily=daily, reset=_reset(resp))
            raise OpenRouterError(resp.status_code, message)
        raise AssertionError("unreachable")

    def _read(self, data: dict, passages: Sequence[Passage]) -> Draft:
        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError(502, "the response had no choices")
        choice = choices[0]
        if choice.get("error"):
            err = choice["error"]
            raise OpenRouterError(int(err.get("code") or 502), str(err.get("message", "")))
        message = choice.get("message") or {}
        if message.get("refusal"):
            return Draft(False, reason=str(message["refusal"]), refused_by_model=True)
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"the answer hit max_tokens={self.max_tokens}; raise it")
        text = message.get("content")
        if not text:
            reason = choice.get("finish_reason")
            raise RuntimeError(f"no text in the response (finish_reason {reason})")
        return parse_answer(extract_json(text), passages)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> str:
    """The JSON object in a reply, without reasoning tags, code fences or prose around it."""
    text = _FENCE.sub("", _THINK.sub("", text).strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
    return text[start : end + 1]


def _json(resp) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": resp.text[:200]}}


def _message(data: dict) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message", "")) or "no message"
    return str(err or "no message")


def _reset(resp) -> float | None:
    """When the rate limit resets, from ``X-RateLimit-Reset`` (milliseconds since the epoch)."""
    try:
        return float(resp.headers.get("x-ratelimit-reset")) / 1000
    except (TypeError, ValueError):
        return None


def _wait(resp, attempt: int) -> float:
    try:
        return min(float(resp.headers.get("retry-after")), 60.0)
    except (TypeError, ValueError):
        return min(5.0 * 2**attempt, 60.0)
