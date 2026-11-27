"""Checks the agent on the real index.

The extractive checks always run in the live job. The Claude checks call the Anthropic
API, so they run only when ANTHROPIC_API_KEY is set (a repository secret in CI) and are
skipped otherwise. Each one is a billed API request. The OpenRouter checks likewise run
only when OPENROUTER_API_KEY is set; they use OPENROUTER_MODEL, or a free model by
default, and are skipped if OpenRouter's daily limit for free models has been reached.
"""

import os
from pathlib import Path

import pytest

from policy_assistant.agent import PolicyAgent
from policy_assistant.answerers import ClaudeAnswerer, ExtractiveAnswerer
from policy_assistant.embed import make_encoder
from policy_assistant.index import PolicyIndex, collection_info, open_client
from policy_assistant.openrouter import OpenRouterAnswerer, RateLimited

from .test_live_retrieval import answers

pytestmark = pytest.mark.live

needs_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"
)

needs_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set"
)

LOST_CARD = (
    "How much can a consumer lose if they report a lost debit card within two business days?"
)

# Questions the default corpus (12 CFR 1002, 1005, 1016, 1030) does not answer.
OUT_OF_SCOPE = [
    "What is the speed limit on interstate highways?",
    "How many weeks of unpaid family leave must an employer give?",
]


@pytest.fixture(scope="module")
def index():
    folder = Path(os.environ.get("GPA_QDRANT", "data/qdrant"))
    if not folder.exists():
        pytest.skip(f"no index at {folder}; run policy-assistant index build first")
    client = open_client(path=str(folder))
    info = collection_info(client)
    if info is None:
        pytest.skip(f"no collection in {folder}")
    yield PolicyIndex(client, make_encoder(info["encoder"]))
    client.close()


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_extractive_refuses_out_of_scope_questions(index, question):
    answer = PolicyAgent(index, ExtractiveAnswerer()).ask(question)
    assert answer.status == "refused", answer.render()


def test_extractive_answers_cite_retrieved_chunks(index):
    answer = PolicyAgent(index, ExtractiveAnswerer()).ask(
        "When must a financial institution provide an initial privacy notice to a customer?"
    )
    retrieved = {p.chunk_id for p in answer.passages}
    assert all(s in retrieved for c in answer.claims for s in c.sources)


@needs_key
def test_claude_answers_with_a_citation_of_the_right_section(index):
    answer = PolicyAgent(index, ClaudeAnswerer()).ask(LOST_CARD)
    print(answer.render())
    assert answer.answered, answer.render()
    assert any(answers(p, "1005.6") for p in answer.sources()), answer.render()
    assert any("$50" in c.text for c in answer.claims), answer.render()


@needs_key
@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_claude_refuses_out_of_scope_questions(index, question):
    answer = PolicyAgent(index, ClaudeAnswerer()).ask(question)
    print(answer.render())
    assert answer.status == "refused", answer.render()


def ask_openrouter(index, question):
    try:
        return PolicyAgent(index, OpenRouterAnswerer()).ask(question)
    except RateLimited as err:
        pytest.skip(f"OpenRouter rate limit: {err}")


@needs_openrouter
def test_openrouter_answers_with_a_citation_of_the_right_section(index):
    answer = ask_openrouter(index, LOST_CARD)
    print(answer.render())
    assert answer.answered, answer.render()
    assert any(answers(p, "1005.6") for p in answer.sources()), answer.render()
    assert any("$50" in c.text for c in answer.claims), answer.render()


@needs_openrouter
def test_openrouter_refuses_an_out_of_scope_question(index):
    # One question, not both: free models allow only a few requests a day.
    answer = ask_openrouter(index, OUT_OF_SCOPE[1])
    print(answer.render())
    assert answer.status == "refused", answer.render()
