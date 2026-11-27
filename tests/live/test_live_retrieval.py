"""Checks the Qdrant index built from the real corpus with the real embedding model.

CI builds the corpus, runs ``policy-assistant index build`` into $GPA_QDRANT, then runs
these with ``pytest -m live``. They are smoke tests: each question below is answered by
one section of the regulation, and that section (or its official interpretation) should
come back in the top five. They are not the evaluation; that is RAGBench's job.
"""

import os
from pathlib import Path

import pytest

from policy_assistant.corpus import load_corpus
from policy_assistant.embed import make_encoder
from policy_assistant.evaluate import answers_section
from policy_assistant.index import PolicyIndex, collection_info, open_client

pytestmark = pytest.mark.live

QUESTIONS = [
    (
        "How much can a consumer lose if they report a lost debit card within two business days?",
        "1005.6",
    ),
    (
        "How long does a bank have to investigate an error a consumer reports about an "
        "electronic fund transfer?",
        "1005.11",
    ),
    (
        "Within how many days must a creditor notify an applicant of action taken on a "
        "completed credit application?",
        "1002.9",
    ),
    (
        "When must a financial institution give a customer its initial privacy notice?",
        "1016.4",
    ),
    (
        "What account disclosures must a depository institution give before an account is opened?",
        "1030.4",
    ),
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


def answers(passage, section: str) -> bool:
    """The passage is from the section, or is an official interpretation of it."""
    return answers_section(passage, section)


def test_every_chunk_is_indexed(index):
    corpus = Path(os.environ.get("GPA_CORPUS", "data/corpus"))
    assert index.count() == len(load_corpus(corpus))


@pytest.mark.parametrize(("question", "section"), QUESTIONS)
def test_the_answering_section_is_in_the_top_five(index, question, section):
    hits = index.search(question, k=5)
    shown = [f"{p.citation} ({p.score:.3f})" for p in hits]
    assert any(answers(p, section) for p in hits), f"{section} not in {shown}"


def test_part_filter_keeps_to_the_part(index):
    hits = index.search("What is an access device?", k=10, parts=["1030"])
    assert hits
    assert {p.part for p in hits} == {"1030"}
