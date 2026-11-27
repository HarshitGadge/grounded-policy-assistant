import json
from types import SimpleNamespace

import pytest

from policy_assistant.agent import PolicyAgent
from policy_assistant.answerers import make_answerer
from policy_assistant.openrouter import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_URL,
    OpenRouterAnswerer,
    OpenRouterError,
    RateLimited,
    extract_json,
)

from .test_answerers import PASSAGES


class Resp:
    def __init__(self, status, data=None, headers=None, text=""):
        self.status_code = status
        self._data = data
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._data is None:
            raise ValueError("not JSON")
        return self._data


class FakeHTTP:
    """Stands in for httpx.Client: records each POST, returns canned responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json, headers):
        self.posts.append(SimpleNamespace(url=url, body=json, headers=headers))
        return self.responses.pop(0)


def ok(content, finish_reason="stop", **message):
    data = content if isinstance(content, str) else json.dumps(content)
    return Resp(
        200,
        {"choices": [{"message": {"content": data, **message}, "finish_reason": finish_reason}]},
    )


ANSWER = {
    "answerable": True,
    "reason": "",
    "claims": [{"text": "Liability is at most $50.", "sources": ["P1"]}],
}


def make(*responses, **kwargs):
    http = FakeHTTP(*responses)
    sleeps = []
    answerer = OpenRouterAnswerer(api_key="k", client=http, sleep=sleeps.append, **kwargs)
    return answerer, http, sleeps


def test_request_shape(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    answerer, http, _ = make(ok(ANSWER))
    draft = answerer.draft("How much is the liability?", PASSAGES, feedback="Try again.")
    assert draft.claims[0].sources == ("c1",)
    post = http.posts[0]
    assert post.url == OPENROUTER_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert post.headers["Authorization"] == "Bearer k"
    body = post.body
    assert body["model"] == DEFAULT_OPENROUTER_MODEL
    assert DEFAULT_OPENROUTER_MODEL.endswith(":free")
    assert body["temperature"] == 0
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"]["sources"][
        "items"
    ]["enum"] == ["P1", "P2"]
    assert body["provider"] == {"require_parameters": True}
    system, user = body["messages"]
    assert "Answer only from the passages" in system["content"]
    assert '<passage label="P1"' in user["content"]
    assert user["content"].endswith("Try again.")


def test_model_from_argument_or_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model:free")
    assert OpenRouterAnswerer(api_key="k", client=FakeHTTP()).model == "some/model:free"
    answerer = OpenRouterAnswerer(model="other/model", api_key="k", client=FakeHTTP())
    assert answerer.name == "openrouter:other/model"


def test_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterAnswerer(client=FakeHTTP())


def test_falls_back_to_json_in_the_prompt_once(monkeypatch):
    no_endpoint = Resp(404, {"error": {"code": 404, "message": "No endpoints found"}})
    reply = ok("Here it is:\n```json\n" + json.dumps(ANSWER) + "\n```")
    answerer, http, _ = make(no_endpoint, reply, ok(ANSWER))
    assert answerer.draft("q", PASSAGES).answerable
    first, second = http.posts[0].body, http.posts[1].body
    assert "response_format" in first
    assert "response_format" not in second and "provider" not in second
    assert "Reply with one JSON object" in second["messages"][0]["content"]
    # It remembers: the next question goes straight to the prompt form.
    answerer.draft("q", PASSAGES)
    assert len(http.posts) == 3
    assert "response_format" not in http.posts[2].body


def test_structured_only_does_not_fall_back():
    answerer, _, _ = make(Resp(404, {"error": {"message": "No endpoints"}}), structured=True)
    with pytest.raises(OpenRouterError) as err:
        answerer.draft("q", PASSAGES)
    assert err.value.status == 404


def test_prompt_only_never_sends_a_schema_format():
    answerer, http, _ = make(ok(ANSWER), structured=False)
    answerer.draft("q", PASSAGES)
    assert "response_format" not in http.posts[0].body


def test_retries_rate_limits_then_gives_up():
    limited = Resp(429, {"error": {"message": "Rate limit exceeded"}}, headers={"retry-after": "2"})
    answerer, http, sleeps = make(limited, limited, ok(ANSWER), retries=3)
    assert answerer.draft("q", PASSAGES).answerable
    assert sleeps == [2.0, 2.0]

    answerer, _, sleeps = make(*[Resp(429, {"error": {"message": "slow down"}})] * 3, retries=2)
    with pytest.raises(RateLimited, match="429"):
        answerer.draft("q", PASSAGES)
    assert sleeps == [5.0, 10.0]


def test_errors_are_reported():
    answerer, _, _ = make(Resp(401, {"error": {"code": 401, "message": "No auth"}}))
    with pytest.raises(OpenRouterError, match="401: No auth"):
        answerer.draft("q", PASSAGES)

    answerer, _, _ = make(Resp(200, {"error": {"code": 502, "message": "upstream"}}))
    with pytest.raises(OpenRouterError, match="upstream"):
        answerer.draft("q", PASSAGES)

    answerer, _, _ = make(Resp(503, None, text="<html>down</html>"), retries=0)
    with pytest.raises(OpenRouterError, match="down"):
        answerer.draft("q", PASSAGES)


def test_refusal_length_and_empty_replies():
    answerer, _, _ = make(ok(None, refusal="I can't help with that."))
    draft = answerer.draft("q", PASSAGES)
    assert draft.refused_by_model and not draft.answerable

    answerer, _, _ = make(ok('{"answerable": tr', finish_reason="length"))
    with pytest.raises(RuntimeError, match="max_tokens"):
        answerer.draft("q", PASSAGES)

    answerer, _, _ = make(ok(""))
    with pytest.raises(RuntimeError, match="no text"):
        answerer.draft("q", PASSAGES)


def test_not_answerable_reply():
    answerer, _, _ = make(ok({"answerable": False, "reason": "not covered", "claims": []}))
    draft = answerer.draft("q", PASSAGES)
    assert (draft.answerable, draft.reason) == (False, "not covered")


def test_extract_json():
    assert extract_json('<think>hmm {x}</think>\n{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": {"b": 2}}\n```') == '{"a": {"b": 2}}'
    assert extract_json('Sure. {"a": 1} Hope that helps.') == '{"a": 1}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_make_answerer_builds_it(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert make_answerer("openrouter").model == DEFAULT_OPENROUTER_MODEL
    named = make_answerer("openrouter:nvidia/nemotron-3.5-super-vl:free")
    assert named.model == "nvidia/nemotron-3.5-super-vl:free"


@pytest.fixture
def small_index(tmp_path):
    from pathlib import Path

    from policy_assistant.corpus import build_corpus, load_corpus
    from policy_assistant.embed import HashingEncoder
    from policy_assistant.index import PolicyIndex, open_client

    xml = (Path(__file__).parent / "fixtures" / "part-9999.xml").read_bytes()
    build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=lambda t, p, d: xml)
    idx = PolicyIndex(open_client(), HashingEncoder(512))
    idx.build(load_corpus(tmp_path))
    return idx


def test_agent_with_openrouter_end_to_end(small_index):
    bad = {"answerable": True, "reason": "", "claims": [{"text": "X.", "sources": ["P9"]}]}
    good = {"answerable": True, "reason": "", "claims": [{"text": "Y.", "sources": ["P1"]}]}
    answerer, http, _ = make(ok(bad), ok(good))
    answer = PolicyAgent(small_index, answerer, k=3).ask("What is a preauthorized transfer?")
    assert answer.answered and answer.attempts == 2
    assert answer.answerer.startswith("openrouter:")
    assert "did not cite" in http.posts[1].body["messages"][1]["content"]
