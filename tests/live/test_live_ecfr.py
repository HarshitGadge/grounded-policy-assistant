"""Checks against the real eCFR text. Run with `pytest -m live`; CI runs them in the live job.

They read the corpus that `policy-assistant corpus build` wrote to $GPA_CORPUS
(default data/corpus), so the build and the checks see the same text.
"""

import os
import re
from collections import Counter
from pathlib import Path

import pytest

from policy_assistant.corpus import load_corpus

pytestmark = pytest.mark.live

SECTION_CITATION = re.compile(r"^12 CFR \d+\.\d+[a-z]?(\([0-9a-zA-Z]+\))*$")


@pytest.fixture(scope="module")
def records():
    folder = Path(os.environ.get("GPA_CORPUS", "data/corpus"))
    if not (folder / "corpus.jsonl").exists():
        pytest.skip(f"no corpus at {folder}; run policy-assistant corpus build first")
    return load_corpus(folder)


def test_every_default_part_is_there(records):
    parts = Counter(r["part"] for r in records)
    for part in ("1002", "1005", "1016", "1030"):
        assert parts[part] >= 20, parts


def test_ids_are_unique_and_text_is_not_empty(records):
    assert len({r["_id"] for r in records}) == len(records)
    assert all(r["text"].strip() for r in records)


def test_section_citations_are_well_formed(records):
    bad = [
        r["citation"]
        for r in records
        if "Appendix" not in r["section"]
        and "Supplement" not in r["section"]
        and not SECTION_CITATION.match(r["citation"])
    ]
    assert not bad, bad[:10]


def test_most_section_chunks_have_paragraph_labels(records):
    sections = [r for r in records if r["section"][0].isdigit()]
    labelled = sum(bool(r["paragraphs"]) for r in sections)
    assert labelled / len(sections) > 0.8


def test_known_paragraphs_are_found(records):
    cites = {r["citation"] for r in records}
    # Reg E's limits on a consumer's liability for unauthorized transfers.
    assert any(c.startswith("12 CFR 1005.6(b)") for c in cites)
    # Reg E's error-resolution timing.
    assert any(c.startswith("12 CFR 1005.11(c)") for c in cites)
    # Reg B's adverse action notices.
    assert any(c.startswith("12 CFR 1002.9(a)") for c in cites)


def test_interpretations_are_cited_as_comments(records):
    comments = [r["citation"] for r in records if "Supplement I" in r["citation"]]
    assert comments
    without = Counter(c for c in comments if ", comment" not in c)
    share = 1 - sum(without.values()) / len(comments)
    assert share > 0.8, without.most_common(25)
