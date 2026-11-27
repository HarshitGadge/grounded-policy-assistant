import json
from types import SimpleNamespace

import pytest

from policy_assistant.answerers import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeAnswerer,
    ExtractiveAnswerer,
    answer_schema,
    content_words,
    make_answerer,
    parse_answer,
    sentences,
)
from policy_assistant.index import Passage


def passage(chunk_id, text, heading="", citation=None):
    return Passage(
        chunk_id=chunk_id,
        score=0.5,
        citation=citation or f"99 CFR {chunk_id}",
        title="",
        text=text,
        part="9999",
        section="9999.3",
        heading=heading,
        date="2026-01-01",
    )


PASSAGES = [
    passage(
        "c1",
        "(b)(1) Timely notice. If the consumer notifies the institution within two business days, liability shall not exceed $50.",
    ),
    passage(
        "c2", "(c) Business day means any day on which the offices of the institution are open."
    ),
]


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned response."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def reply(data, stop_reason="end_turn"):
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=json.dumps(data)),
    ]
    return SimpleNamespace(stop_reason=stop_reason, content=content, stop_details=None)


def test_claude_request_shape():
    answerer = ClaudeAnswerer(client=FakeClient())
    req = answerer.request("How much is the liability?", PASSAGES, feedback="Try again.")
    assert req["model"] == DEFAULT_CLAUDE_MODEL == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "medium"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert req["fallbacks"] == "default"
    prompt = req["messages"][0]["content"]
    assert '<passage label="P1" citation="99 CFR c1">' in prompt
    assert prompt.index("P2") > prompt.index("P1")
    assert "Question: How much is the liability?" in prompt
    assert prompt.endswith("Try again.")
    assert "Answer only from the passages" in req["system"]


def test_claude_request_without_fallbacks():
    req = ClaudeAnswerer(client=FakeClient(), fallbacks=None).request("q", PASSAGES)
    assert "fallbacks" not in req and "betas" not in req


def test_schema_only_allows_given_labels():
    schema = answer_schema(3)
    item = schema["properties"]["claims"]["items"]["properties"]["sources"]["items"]
    assert item["enum"] == ["P1", "P2", "P3"]
    assert schema["additionalProperties"] is False


def test_claude_answer_maps_labels_to_chunk_ids():
    client = FakeClient(
        reply(
            {
                "answerable": True,
                "reason": "",
                "claims": [{"text": "Liability is capped at $50.", "sources": ["P1", "P2"]}],
            }
        )
    )
    draft = ClaudeAnswerer(client=client).draft("How much?", PASSAGES)
    assert draft.answerable
    assert draft.claims[0].sources == ("c1", "c2")
    assert len(client.requests) == 1


def test_claude_refusal_to_answer():
    client = FakeClient(reply({"answerable": False, "reason": "Not covered.", "claims": []}))
    draft = ClaudeAnswerer(client=client).draft("Minimum wage?", PASSAGES)
    assert (draft.answerable, draft.reason, draft.refused_by_model) == (
        False,
        "Not covered.",
        False,
    )


def test_claude_safety_refusal():
    response = SimpleNamespace(
        stop_reason="refusal", content=[], stop_details=SimpleNamespace(category="cyber")
    )
    draft = ClaudeAnswerer(client=FakeClient(response)).draft("q", PASSAGES)
    assert draft.refused_by_model
    assert draft.reason == "category: cyber"


def test_claude_max_tokens_is_an_error():
    with pytest.raises(RuntimeError, match="max_tokens"):
        ClaudeAnswerer(client=FakeClient(reply({}, stop_reason="max_tokens"))).draft("q", PASSAGES)


def test_claude_no_text_is_an_error():
    empty = SimpleNamespace(stop_reason="end_turn", content=[], stop_details=None)
    with pytest.raises(RuntimeError, match="no text"):
        ClaudeAnswerer(client=FakeClient(empty)).draft("q", PASSAGES)


def test_parse_answer_keeps_unknown_labels_for_the_check():
    draft = parse_answer(
        json.dumps(
            {"answerable": True, "reason": "", "claims": [{"text": "x", "sources": ["P9"]}]}
        ),
        PASSAGES,
    )
    assert draft.claims[0].sources == ("P9",)


def test_parse_answer_answerable_without_claims_is_not_an_answer():
    draft = parse_answer(json.dumps({"answerable": True, "reason": "", "claims": []}), PASSAGES)
    assert not draft.answerable
    assert draft.reason == "the model returned no claims"


def test_default_client_is_the_anthropic_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    import anthropic

    assert isinstance(ClaudeAnswerer().client, anthropic.Anthropic)


def test_content_words_and_sentences():
    assert content_words("What are the Business days?") == {"business", "day"}
    assert content_words("the process") == {"process"}
    assert sentences("One. (b) Two; Three: four.") == ["One.", "(b) Two;", "Three: four."]


def test_extractive_quotes_the_best_sentence():
    draft = ExtractiveAnswerer().draft("What does business day mean?", PASSAGES)
    assert draft.answerable
    assert draft.claims[0].sources == ("c2",)
    assert draft.claims[0].text.startswith("(c) Business day means")


def test_extractive_uses_the_heading_as_context():
    p = passage(
        "c3", "The two days begin when the consumer learns of the loss.", heading="Timely notice"
    )
    draft = ExtractiveAnswerer().draft("When does timely notice begin?", [p])
    assert draft.answerable


def test_extractive_refuses_off_topic_questions():
    draft = ExtractiveAnswerer().draft("What is the capital of France?", PASSAGES)
    assert not draft.answerable
    assert "covers enough" in draft.reason


def test_extractive_refuses_a_question_of_stopwords():
    assert not ExtractiveAnswerer().draft("What is it?", PASSAGES).answerable


def test_extractive_takes_one_sentence_per_passage():
    draft = ExtractiveAnswerer(min_coverage=0.3, max_claims=3).draft(
        "Which institution business day notice?", PASSAGES
    )
    assert [c.sources[0] for c in draft.claims] == ["c1", "c2"]


def test_extractive_rejects_bad_coverage():
    with pytest.raises(ValueError):
        ExtractiveAnswerer(min_coverage=0)


def test_make_answerer(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    assert isinstance(make_answerer("extractive"), ExtractiveAnswerer)
    assert make_answerer("claude").model == "claude-opus-5-5"
    assert make_answerer("claude:claude-sonnet-5-5").name == "claude:claude-sonnet-5-5"
    assert make_answerer("claude", model="claude-haiku-5-5", effort="low").effort == "low"
    with pytest.raises(ValueError):
        make_answerer("gpt")
