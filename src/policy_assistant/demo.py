"""What the Streamlit landing page (``streamlit_app.py``) shows, kept apart from the UI so
it can be tested.

The page builds its own index on first start: it downloads the four default parts from
eCFR as of :data:`CORPUS_DATE` (the date the model scores in the README were run on),
embeds them with the default encoder into an in-memory Qdrant collection, and answers
with the extractive answerer, which needs no key. With ``OPENROUTER_API_KEY`` set (a
Streamlit secret), visitors can also pick the free OpenRouter model.

:data:`RESULTS` repeats figures from the README's Results section, each with the CI run
it came from; a test checks that every one of them appears there.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .agent import Answer
from .corpus import DEFAULT_PARTS, build_corpus, load_corpus
from .embed import DEFAULT_MODEL, make_encoder
from .index import Passage, PolicyIndex, open_client

CORPUS_DATE = "2026-11-01"
REPO = "https://github.com/HarshitGadge/grounded-policy-assistant"
RAGBENCH = "https://github.com/HarshitGadge/ragbench"
BOOTH = "https://harshitgadge.github.io/"


def run_url(run_id: str) -> str:
    return f"{REPO}/actions/runs/{run_id}"


@dataclass(frozen=True)
class Figure:
    label: str
    value: str
    detail: str
    run: str


RESULTS: tuple[Figure, ...] = (
    Figure(
        "In-scope questions answered correctly",
        "80%",
        "39 of 49, free model on OpenRouter (nemotron-3.5-super-vl), k = 5",
        "46050443255",
    ),
    Figure(
        "Out-of-scope questions refused",
        "92%",
        "11 of 12, same runs",
        "46050443255",
    ),
    Figure(
        "Correct when it answers",
        "100%",
        "every answer cited the labeled section",
        "46050443255",
    ),
    Figure(
        "Retrieval nDCG@10, Qdrant vs BM25",
        "0.5894",
        "BM25 scores 0.4505; scored by RAGBench on 49 questions",
        "46043610493",
    ),
)

REGULATIONS: tuple[tuple[str, str, str], ...] = (
    ("12 CFR 1002", "Regulation B", "Equal credit opportunity"),
    ("12 CFR 1005", "Regulation E", "Electronic fund transfers, card and transfer errors"),
    ("12 CFR 1016", "Regulation P", "Privacy of consumer financial information"),
    ("12 CFR 1030", "Regulation DD", "Truth in savings"),
)

# The last one is out of scope, to show a refusal.
EXAMPLES: tuple[str, ...] = (
    "How much can a consumer lose if they report a lost debit card within two business days?",
    "Can a creditor ask about an applicant's marital status?",
    "What is a preauthorized transfer?",
    "When must a financial institution give a customer an annual privacy notice?",
    "Which credit card has the best travel rewards?",
)

GRAPH = """digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname="Helvetica"];
  retrieve -> grade;
  grade -> refuse [label="nothing relevant"];
  grade -> generate;
  generate -> verify;
  verify -> answer [label="every claim cited"];
  verify -> generate [label="a claim lost its source"];
  verify -> refuse [label="the text doesn't answer"];
  answer [style="rounded,bold"]; refuse [style="rounded,dashed"];
}"""


def build_index(
    corpus_dir: str | Path,
    date: str = CORPUS_DATE,
    encoder: str = DEFAULT_MODEL,
    parts: Iterable[tuple[int, str]] = DEFAULT_PARTS,
    fetch=None,
) -> PolicyIndex:
    """An in-memory index of the corpus in ``corpus_dir``, building the corpus there first
    (from eCFR, or with ``fetch``) if it isn't built yet."""
    folder = Path(corpus_dir)
    if not (folder / "corpus.jsonl").exists():
        build_corpus(folder, date, parts, fetch=fetch)
    index = PolicyIndex(open_client(), make_encoder(encoder))
    index.build(load_corpus(folder), metadata={"corpus_date": date})
    return index


def answerer_choices(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Label -> answerer name. The model is offered only when its key is set."""
    env = os.environ if env is None else env
    choices = {"Extractive baseline (no model)": "extractive"}
    if env.get("OPENROUTER_API_KEY"):
        choices["Free model on OpenRouter"] = "openrouter"
    return choices


_SECTION = re.compile(r"^\d+\.\d+$")


def ecfr_link(passage: Passage) -> str:
    """The passage's section on eCFR as of the corpus date, or its part for a supplement
    or appendix."""
    title = passage.citation.split(" ", 1)[0]
    where = f"section-{passage.section}" if _SECTION.match(passage.section) else ""
    where = where or f"part-{passage.part}"
    date = passage.date or CORPUS_DATE
    return f"https://www.ecfr.gov/on/{date}/title-{title}/{where}"


REFUSAL_LABEL = {
    "no_passages": "Nothing in the four regulations is close enough to this question.",
    "unsupported": "The retrieved regulation text doesn't answer this question.",
    "uncited": "No statement in the draft answer was backed by a retrieved passage.",
    "model_refused": "The model declined this request.",
}


def view(answer: Answer) -> dict:
    """What the page shows for an answer: numbered claims and their sources, or why it
    refused and the closest passages."""
    passage = {
        "citation": lambda p: p.citation,
        "text": lambda p: p.text,
        "score": lambda p: round(p.score, 3),
        "link": ecfr_link,
    }

    def show(p: Passage) -> dict:
        return {k: f(p) for k, f in passage.items()}

    if not answer.answered:
        return {
            "answered": False,
            "why": REFUSAL_LABEL.get(answer.reason, answer.reason),
            "detail": answer.detail,
            "closest": [show(p) for p in answer.passages[:3]],
        }
    sources = answer.sources()
    number = {p.chunk_id: i for i, p in enumerate(sources, 1)}
    return {
        "answered": True,
        "claims": [
            {"text": c.text, "marks": [number[s] for s in c.sources if s in number]}
            for c in answer.claims
        ],
        "sources": [{"n": number[p.chunk_id], **show(p)} for p in sources],
    }
