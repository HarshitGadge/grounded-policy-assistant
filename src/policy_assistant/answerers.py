"""Answerers: they turn a question and its retrieved passages into cited claims.

* :class:`ClaudeAnswerer` asks Claude, through the Anthropic API, for a structured answer:
  either claims that each cite the passages they rest on, or a refusal with a reason.
  It needs an Anthropic API key (``ANTHROPIC_API_KEY``), and each question is a billed
  API request.
* :class:`ExtractiveAnswerer` quotes the passage sentence that covers the most of the
  question's words, or refuses when no sentence covers enough of them. It calls no
  model, so it is free and deterministic: a baseline, and what the tests use.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .agent import Claim, Draft
from .index import Passage

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You answer questions about U.S. federal regulations for people who need to rely on the \
answer. You are given numbered passages from the Code of Federal Regulations and its \
official interpretations, retrieved for the question. They are your only source.

Answer only from the passages. Break the answer into short claims. Each claim cites, by \
label, every passage it rests on, and says nothing those passages don't say. Keep \
numbers, deadlines, dollar amounts and conditions exactly as the passages state them.

If the passages don't answer the question, or only answer part of it in a way that \
would mislead, set answerable to false, give the reason in one sentence, and return no \
claims. Don't fill gaps from your own knowledge, even when you know the answer: a \
refusal is the right result when the text isn't there."""


def _labels(n: int) -> list[str]:
    return [f"P{i}" for i in range(1, n + 1)]


def answer_schema(n_passages: int) -> dict:
    """The JSON schema of Claude's answer. Citations can only name passages it was given."""
    return {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "reason": {
                "type": "string",
                "description": "If not answerable, why, in one sentence. Otherwise empty.",
            },
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": _labels(n_passages)},
                        },
                    },
                    "required": ["text", "sources"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answerable", "reason", "claims"],
        "additionalProperties": False,
    }


def format_passages(passages: Sequence[Passage]) -> str:
    blocks = []
    for label, p in zip(_labels(len(passages)), passages, strict=True):
        blocks.append(f'<passage label="{label}" citation="{p.citation}">\n{p.text}\n</passage>')
    return "\n\n".join(blocks)


class ClaudeAnswerer:
    """Claude answers from the passages with structured output.

    The schema only lets a citation name a passage label it was shown, and the agent
    still checks every citation. A request the model declines comes back as a refusal.
    ``fallbacks="default"`` lets the API retry a declined request on Anthropic's
    recommended fallback model; pass ``fallbacks=None`` to turn that off.
    """

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        client: Any = None,
        effort: str = "medium",
        max_tokens: int = 16000,
        fallbacks: str | None = "default",
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.name = f"claude:{model}"
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks

    def request(self, question: str, passages: Sequence[Passage], feedback: str = "") -> dict:
        """The keyword arguments sent to ``client.beta.messages.create``."""
        prompt = f"{format_passages(passages)}\n\nQuestion: {question}"
        if feedback:
            prompt += f"\n\n{feedback}"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": answer_schema(len(passages))},
            },
        }
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = self.fallbacks
        return kwargs

    def draft(self, question: str, passages: Sequence[Passage], feedback: str = "") -> Draft:
        response = self.client.beta.messages.create(**self.request(question, passages, feedback))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return Draft(False, reason=f"category: {category}", refused_by_model=True)
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"the answer hit max_tokens={self.max_tokens}; raise it")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text in the response (stop_reason {response.stop_reason})")
        return parse_answer(text, passages)


def parse_answer(text: str, passages: Sequence[Passage]) -> Draft:
    """Read Claude's JSON answer and map passage labels back to chunk ids.

    A label that names no passage is kept as written, so the agent's check drops it.
    """
    data = json.loads(text)
    by_label = dict(zip(_labels(len(passages)), (p.chunk_id for p in passages), strict=True))
    claims = tuple(
        Claim(str(c.get("text", "")), tuple(by_label.get(s, s) for s in c.get("sources", [])))
        for c in data.get("claims", [])
    )
    answerable = bool(data.get("answerable")) and bool(claims)
    reason = str(data.get("reason", ""))
    if data.get("answerable") and not claims:
        reason = reason or "the model returned no claims"
    return Draft(answerable, claims if answerable else (), reason)


# -- extractive baseline -------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9$]+(?:\.[0-9]+)?")
_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[(A-Z0-9])")
_STOPWORD_TEXT = """
a an and are as at be been by can could do does for from had has have how i if in
into is it its may me must my of on or our shall should so such than that the their
them then there these they this to under was we were what when where which who whom
why will with would you your about any each other
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def content_words(text: str) -> set[str]:
    words = set()
    for w in _WORD.findall(text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def sentences(text: str) -> list[str]:
    """Split at line breaks (chunks keep one paragraph per line) and sentence ends."""
    return [s.strip() for line in text.splitlines() for s in _SENTENCE.split(line) if s.strip()]


class ExtractiveAnswerer:
    """Quotes the passage sentence that covers the most of the question's content words.

    It answers when the best sentence covers at least ``min_coverage`` of them, and adds
    up to ``max_claims - 1`` more sentences from other passages that clear the same bar.
    Otherwise it refuses. It understands nothing: it is the floor any model should beat.
    """

    name = "extractive"

    def __init__(self, min_coverage: float = 0.6, max_claims: int = 2):
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        self.min_coverage = min_coverage
        self.max_claims = max_claims

    def draft(self, question: str, passages: Sequence[Passage], feedback: str = "") -> Draft:
        wanted = content_words(question)
        if not wanted:
            return Draft(False, reason="the question has no content words to match")
        scored = []
        for rank, p in enumerate(passages):
            context = content_words(p.heading)
            for s in sentences(p.text):
                covered = len(wanted & (content_words(s) | context)) / len(wanted)
                scored.append((covered, -rank, s, p.chunk_id))
        scored.sort(reverse=True)
        claims: list[Claim] = []
        used: set[str] = set()
        for covered, _, sentence, chunk_id in scored:
            if covered < self.min_coverage or len(claims) == self.max_claims:
                break
            if chunk_id in used:
                continue
            used.add(chunk_id)
            claims.append(Claim(sentence, (chunk_id,)))
        if not claims:
            best = scored[0][0] if scored else 0.0
            return Draft(
                False,
                reason=f"no passage sentence covers enough of the question ({best:.0%})",
            )
        return Draft(True, tuple(claims))


def make_answerer(name: str, **kwargs: Any):
    """``extractive`` or ``claude`` (optionally ``claude:<model id>``)."""
    if name == "extractive":
        return ExtractiveAnswerer(**kwargs)
    if name == "claude" or name.startswith("claude:"):
        model = name.partition(":")[2] or kwargs.pop("model", DEFAULT_CLAUDE_MODEL)
        kwargs.pop("model", None)
        return ClaudeAnswerer(model=model, **kwargs)
    raise ValueError(f"unknown answerer {name!r}; use extractive or claude")
