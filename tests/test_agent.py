from pathlib import Path

import pytest

from policy_assistant.agent import (
    MODEL_REFUSED,
    NO_PASSAGES,
    UNCITED,
    UNSUPPORTED,
    Answer,
    Claim,
    Draft,
    PolicyAgent,
)
from policy_assistant.answerers import ExtractiveAnswerer
from policy_assistant.corpus import build_corpus, load_corpus
from policy_assistant.embed import HashingEncoder
from policy_assistant.index import PolicyIndex, open_client


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    folder = tmp_path_factory.mktemp("corpus")
    xml = (Path(__file__).parent / "fixtures" / "part-9999.xml").read_bytes()
    build_corpus(folder, "2026-01-01", [(99, "9999")], fetch=lambda t, p, d: xml)
    return load_corpus(folder)


@pytest.fixture(scope="module")
def index(records):
    idx = PolicyIndex(open_client(), HashingEncoder(512))
    idx.build(records)
    return idx


@pytest.fixture(scope="module")
def ids(records):
    return {r["citation"]: r["_id"] for r in records}


class Scripted:
    """An answerer that returns prepared drafts in turn and records what it was shown."""

    name = "scripted"

    def __init__(self, *drafts):
        self.drafts = list(drafts)
        self.calls = []

    def draft(self, question, passages, feedback=""):
        self.calls.append({"question": question, "passages": list(passages), "feedback": feedback})
        made = self.drafts.pop(0)
        return made(passages) if callable(made) else made


def cite_top(text="A preauthorized transfer recurs at regular intervals."):
    return lambda passages: Draft(True, (Claim(text, (passages[0].chunk_id,)),))


def test_answers_with_citations_of_retrieved_chunks(index, ids):
    answerer = Scripted(cite_top())
    answer = PolicyAgent(index, answerer, k=3).ask("What is a preauthorized transfer?")
    assert answer.answered
    assert answer.attempts == 1
    assert answer.claims[0].sources == (ids["99 CFR 9999.2(i)"],)
    assert [p.citation for p in answer.sources()] == ["99 CFR 9999.2(i)"]
    assert len(answerer.calls[0]["passages"]) == 3
    assert answer.answerer == "scripted"


def test_refuses_when_the_answerer_says_the_text_does_not_answer(index):
    answerer = Scripted(Draft(False, reason="the passages don't mention France"))
    answer = PolicyAgent(index, answerer).ask("What is the capital of France?")
    assert answer.status == "refused"
    assert answer.reason == UNSUPPORTED
    assert answer.detail == "the passages don't mention France"
    assert answer.claims == ()
    assert answer.passages  # what it looked at is kept, to explain the refusal


def test_refuses_without_asking_when_nothing_scores_high_enough(index):
    answerer = Scripted()
    answer = PolicyAgent(index, answerer, min_score=2.0).ask("What is a preauthorized transfer?")
    assert (answer.status, answer.reason) == ("refused", NO_PASSAGES)
    assert answerer.calls == []
    assert answer.passages == ()


def test_a_model_refusal_is_a_refusal(index):
    answerer = Scripted(Draft(False, reason="category: cyber", refused_by_model=True))
    answer = PolicyAgent(index, answerer).ask("What is an access device?")
    assert (answer.status, answer.reason) == ("refused", MODEL_REFUSED)


def test_drops_a_claim_citing_an_unretrieved_chunk_and_asks_again(index, ids):
    bad = Draft(True, (Claim("Made up.", ("not-a-chunk",)),))
    answerer = Scripted(bad, cite_top())
    answer = PolicyAgent(index, answerer).ask("What is a preauthorized transfer?")
    assert answer.answered
    assert answer.attempts == 2
    assert "'Made up.'" in answerer.calls[1]["feedback"]
    assert all(s in {p.chunk_id for p in answer.passages} for c in answer.claims for s in c.sources)


def test_keeps_only_cited_claims_when_out_of_attempts(index):
    mixed = lambda ps: Draft(  # noqa: E731
        True, (Claim("Cited.", (ps[0].chunk_id,)), Claim("Uncited.", ()))
    )
    answer = PolicyAgent(index, Scripted(mixed), max_attempts=1).ask("What is a consumer?")
    assert answer.answered
    assert [c.text for c in answer.claims] == ["Cited."]


def test_refuses_when_no_claim_is_ever_cited(index):
    uncited = Draft(True, (Claim("Uncited.", ("nope",)),))
    answerer = Scripted(uncited, uncited)
    answer = PolicyAgent(index, answerer, max_attempts=2).ask("What is a consumer?")
    assert (answer.status, answer.reason, answer.attempts) == ("refused", UNCITED, 2)


def test_duplicate_and_unknown_sources_are_cleaned(index):
    def messy(ps):
        a = ps[0].chunk_id
        return Draft(True, (Claim("  Text.  ", (a, "ghost", a)),))

    answer = PolicyAgent(index, Scripted(messy), max_attempts=1).ask("What is a consumer?")
    assert answer.claims == (Claim("Text.", (answer.passages[0].chunk_id,)),)


def test_part_filter_reaches_retrieval(index):
    answerer = Scripted(cite_top())
    PolicyAgent(index, answerer).ask("What is a consumer?", parts=["1005"])
    # The fixture has only part 9999, so nothing is retrieved and the model isn't asked.
    assert answerer.calls == []


def test_max_attempts_must_be_positive(index):
    with pytest.raises(ValueError):
        PolicyAgent(index, Scripted(), max_attempts=0)


def test_render_and_to_dict(index, ids):
    def two(ps):
        a, b = ps[0].chunk_id, ps[1].chunk_id
        return Draft(True, (Claim("First.", (a,)), Claim("Second.", (b, a))))

    answer = PolicyAgent(index, Scripted(two), k=2).ask("What is a preauthorized transfer?")
    text = answer.render()
    assert "First. [1]" in text
    assert "Second. [2][1]" in text
    assert "[1] 99 CFR 9999.2(i)" in text
    data = answer.to_dict()
    assert data["status"] == "answered"
    assert data["claims"][0]["citations"] == ["99 CFR 9999.2(i)"]
    assert len(data["passages"]) == 2


def test_render_refusal_lists_closest_passages(index):
    answer = PolicyAgent(index, Scripted(Draft(False))).ask("What is a preauthorized transfer?")
    text = answer.render()
    assert text.startswith("Refused (unsupported): the retrieved regulation text doesn't answer")
    assert "Closest passages:" in text
    assert "- 99 CFR 9999.2(i)" in text


def test_answer_without_passages_renders():
    answer = Answer("q", "refused", (), (), reason=NO_PASSAGES)
    assert answer.render() == (
        "Refused (no_passages): nothing in the index is close enough to this question."
    )


def test_end_to_end_with_the_extractive_answerer(index, ids):
    agent = PolicyAgent(index, ExtractiveAnswerer(), k=5)
    answer = agent.ask("What is a preauthorized transfer?")
    assert answer.answered
    assert answer.claims[0].sources == (ids["99 CFR 9999.2(i)"],)
    assert answer.claims[0].text.startswith("(i) Preauthorized transfer means")

    refused = agent.ask("What is the capital of France?")
    assert refused.status == "refused"


def test_the_graph_has_the_documented_nodes(index):
    nodes = set(PolicyAgent(index, Scripted()).graph.get_graph().nodes)
    assert {"retrieve", "grade", "generate", "verify"} <= nodes
