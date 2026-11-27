"""The evaluation set, as a BEIR dataset for RAGBench, and the agent's own evaluation.

``data/questions.jsonl`` holds hand-written questions. Each in-scope question is labeled
with the one CFR section that answers it (for example ``1005.6``); out-of-scope
questions have no section, because nothing in the corpus answers them.

* :func:`build_eval` turns the labels into relevance judgements over a built corpus:
  every chunk of the labeled section, and every official interpretation (Supplement I
  comment) of it, is relevant. It writes a BEIR folder that ``ragbench run`` scores.
  Out-of-scope questions are left out of it, since they have no relevant chunk.
* :func:`evaluate_agent` asks the agent every question and counts how often it answers
  in-scope questions citing a relevant chunk, and how often it refuses the others.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from .corpus import load_corpus


def load_questions(path: str | Path | None = None) -> list[dict]:
    """The question set: ``_id``, ``text`` and ``section`` (None when out of scope)."""
    if path is None:
        text = resources.files("policy_assistant").joinpath("data/questions.jsonl").read_text()
    else:
        text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def answers_section(record: Mapping[str, Any], section: str) -> bool:
    """The chunk is from the section, or is an official interpretation of it.

    ``record`` is a corpus record or anything with ``section`` and ``citation``.
    """
    get = record.get if isinstance(record, Mapping) else lambda k: getattr(record, k)
    if get("section") == section:
        return True
    part, _, number = section.partition(".")
    return get("section") == f"Supplement I to Part {part}" and bool(
        re.search(rf"\bcomments? {re.escape(number)}\(", get("citation") or "")
    )


def qrels(records: Sequence[Mapping], questions: Iterable[dict]) -> dict[str, list[str]]:
    """For each in-scope question, the ids of the chunks that answer it."""
    out = {}
    for q in questions:
        if q.get("section"):
            out[q["_id"]] = [r["_id"] for r in records if answers_section(r, q["section"])]
    return out


def build_eval(corpus: str | Path, out: str | Path, questions: Sequence[dict] | None = None):
    """Write a BEIR folder (corpus, in-scope queries, qrels) and return a summary.

    Raises ValueError if a labeled section has no chunk in the corpus, since a question
    with no relevant chunk would be scored as a miss that no retriever could avoid.
    """
    questions = list(questions if questions is not None else load_questions())
    records = load_corpus(corpus)
    judged = qrels(records, questions)
    empty = sorted({q["section"] for q in questions if q["_id"] in judged and not judged[q["_id"]]})
    if empty:
        raise ValueError(f"no chunk in {corpus} answers section(s) {', '.join(empty)}")
    out = Path(out)
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(corpus) / "corpus.jsonl", out / "corpus.jsonl")
    with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            if q["_id"] in judged:
                fh.write(json.dumps({"_id": q["_id"], "text": q["text"]}) + "\n")
    with (out / "qrels" / "test.tsv").open("w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for qid, ids in judged.items():
            fh.writelines(f"{qid}\t{doc}\t1\n" for doc in ids)
    sizes = [len(ids) for ids in judged.values()]
    return {
        "queries": len(judged),
        "out_of_scope": len(questions) - len(judged),
        "sections": len({q["section"] for q in questions if q.get("section")}),
        "judgements": sum(sizes),
        "relevant_per_query": {"min": min(sizes), "max": max(sizes)} if sizes else {},
    }


def evaluate_agent(
    agent,
    questions: Sequence[dict],
    progress=None,
    *,
    done: Sequence[dict] = (),
    on_row: Callable[[dict], None] | None = None,
    can_ask: Callable[[], bool] | None = None,
    errors: tuple[type[BaseException], ...] = (),
    halt: Callable[[BaseException], bool] | None = None,
) -> dict:
    """Ask the agent every question; score answers and refusals.

    For an in-scope question, the agent is *correct* when it answers and at least one
    cited chunk answers the labeled section. For an out-of-scope question, it is correct
    when it refuses.

    A run can stop partway and be picked up later:

    * ``done`` holds rows already scored (from an earlier run); those questions are not
      asked again.
    * ``on_row(result)`` is called with the result so far after each new row, so it can
      be saved as the run goes.
    * ``can_ask()`` is checked before each question; when it returns False the run stops,
      for example because a request budget is spent.
    * An exception in ``errors`` raised while asking is recorded as an ``error`` row that
      counts as wrong, unless ``halt(exc)`` is true: then the run stops without a row for
      that question, so a later run asks it again.

    The result says whether every question was scored (``complete``) and, if not, why it
    stopped (``stopped``).
    """
    ids = {q["_id"] for q in questions}
    rows = {r["_id"]: r for r in done if r["_id"] in ids}
    stopped = None
    todo = [q for q in questions if q["_id"] not in rows]
    for i, q in enumerate(todo, 1):
        if can_ask is not None and not can_ask():
            stopped = "the request budget is spent"
            break
        t0 = time.perf_counter()
        try:
            answer = agent.ask(q["text"])
        except errors as err:
            if halt is not None and halt(err):
                stopped = str(err)
                break
            rows[q["_id"]] = _error_row(q, err, time.perf_counter() - t0)
        else:
            rows[q["_id"]] = _row(q, answer, time.perf_counter() - t0)
        if on_row is not None:
            on_row(_result(questions, rows, None))
        if progress:
            progress(i, len(todo))
    return _result(questions, rows, stopped)


def _row(q: dict, answer, seconds: float) -> dict:
    section = q.get("section")
    cited = answer.sources()
    row = {
        "_id": q["_id"],
        "question": q["text"],
        "section": section,
        "status": answer.status,
        "reason": answer.reason,
        "attempts": answer.attempts,
        "citations": [p.citation for p in cited],
        "retrieved_relevant": bool(section)
        and any(answers_section(p, section) for p in answer.passages),
        "seconds": round(seconds, 3),
    }
    if section:
        row["correct"] = answer.answered and any(answers_section(p, section) for p in cited)
    else:
        row["correct"] = not answer.answered
    return row


def _error_row(q: dict, err: BaseException, seconds: float) -> dict:
    """A question the answerer failed on (a malformed reply, say). It counts as wrong."""
    return {
        "_id": q["_id"],
        "question": q["text"],
        "section": q.get("section"),
        "status": "error",
        "reason": f"{type(err).__name__}: {err}"[:300],
        "attempts": 0,
        "citations": [],
        "retrieved_relevant": False,
        "seconds": round(seconds, 3),
        "correct": False,
    }


def _result(questions: Sequence[dict], rows: Mapping[str, dict], stopped) -> dict:
    ordered = [rows[q["_id"]] for q in questions if q["_id"] in rows]
    complete = len(ordered) == len(questions)
    return {
        "summary": summarize(ordered),
        "complete": complete,
        "scored": len(ordered),
        "total": len(questions),
        "stopped": None if complete else stopped,
        "questions": ordered,
    }


def summarize(rows: Sequence[dict]) -> dict:
    inscope = [r for r in rows if r["section"]]
    oos = [r for r in rows if not r["section"]]
    answered = [r for r in inscope if r["status"] == "answered"]

    def rate(part, whole):
        return round(len(part) / len(whole), 4) if whole else None

    return {
        "in_scope": len(inscope),
        "out_of_scope": len(oos),
        "answered_in_scope": rate(answered, inscope),
        "correct_in_scope": rate([r for r in inscope if r["correct"]], inscope),
        "correct_when_answered": rate([r for r in answered if r["correct"]], answered),
        "relevant_retrieved": rate([r for r in inscope if r["retrieved_relevant"]], inscope),
        "refused_out_of_scope": rate([r for r in oos if r["correct"]], oos),
        "errors": sum(r["status"] == "error" for r in rows),
    }


def markdown_summary(results: Sequence[dict]) -> str:
    """A table of agent results, one row per answerer, for the README."""
    header = (
        "| answerer | k | in-scope correct | answered | correct when answered "
        "| out-of-scope refused |\n|---|---|---|---|---|---|\n"
    )
    lines = []
    for res in results:
        s = res["summary"]
        name = res["answerer"]
        if res.get("complete") is False:
            name += f" (partial, {res['scored']} of {res['total']})"
        lines.append(
            f"| {name} | {res['k']} | {_pct(s['correct_in_scope'])} "
            f"| {_pct(s['answered_in_scope'])} | {_pct(s['correct_when_answered'])} "
            f"| {_pct(s['refused_out_of_scope'])} |"
        )
    n = results[0]["summary"] if results else {"in_scope": 0, "out_of_scope": 0}
    note = f"\n{n['in_scope']} in-scope and {n['out_of_scope']} out-of-scope questions.\n"
    return header + "\n".join(lines) + "\n" + note


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"
