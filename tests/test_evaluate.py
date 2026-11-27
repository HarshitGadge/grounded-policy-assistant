import json
from pathlib import Path

import pytest

from policy_assistant import cli
from policy_assistant.agent import Claim, Draft, PolicyAgent
from policy_assistant.answerers import ExtractiveAnswerer
from policy_assistant.corpus import build_corpus, load_corpus
from policy_assistant.embed import HashingEncoder
from policy_assistant.evaluate import (
    answers_section,
    build_eval,
    evaluate_agent,
    load_questions,
    markdown_summary,
    qrels,
    summarize,
)
from policy_assistant.index import PolicyIndex, open_client

ragbench = pytest.importorskip("ragbench")

QUESTIONS = [
    {"_id": "q1", "text": "What is a preauthorized transfer?", "section": "9999.2"},
    {"_id": "q2", "text": "Which error resolution procedures must be kept?", "section": "9999.3"},
    {"_id": "oos1", "text": "What is the capital of France?", "section": None},
]


@pytest.fixture
def corpus(tmp_path):
    xml = (Path(__file__).parent / "fixtures" / "part-9999.xml").read_bytes()
    build_corpus(tmp_path / "corpus", "2026-01-01", [(99, "9999")], fetch=lambda t, p, d: xml)
    return tmp_path / "corpus"


def test_the_built_in_questions_are_well_formed():
    questions = load_questions()
    ids = [q["_id"] for q in questions]
    assert len(ids) == len(set(ids))
    inscope = [q for q in questions if q["section"]]
    assert len(inscope) >= 40 and len(questions) - len(inscope) >= 10
    parts = {q["section"].split(".")[0] for q in inscope}
    assert parts == {"1002", "1005", "1016", "1030"}  # the default corpus
    assert all(q["text"].endswith("?") for q in questions)


def test_answers_section_takes_interpretations():
    assert answers_section({"section": "1005.6", "citation": "12 CFR 1005.6(b)(1)"}, "1005.6")
    comment = {
        "section": "Supplement I to Part 1005",
        "citation": "12 CFR part 1005, Supplement I, comment 6(b)-1",
    }
    assert answers_section(comment, "1005.6")
    assert not answers_section(comment, "1005.7")
    assert not answers_section({"section": "1005.60", "citation": "x"}, "1005.6")


def test_build_eval_writes_a_beir_folder_ragbench_loads(corpus, tmp_path):
    summary = build_eval(corpus, tmp_path / "eval", QUESTIONS)
    assert summary["queries"] == 2 and summary["out_of_scope"] == 1
    data = ragbench.load_beir(tmp_path / "eval")
    assert set(data.queries) == {"q1", "q2"}
    records = load_corpus(corpus)
    assert data.relevant("q1") == set(qrels(records, QUESTIONS)["q1"])
    citations = {r["citation"] for r in records if r["_id"] in data.relevant("q1")}
    assert "99 CFR 9999.2(i)" in citations
    assert "99 CFR part 9999, Supplement I to Part 9999, comment 2(b)(1)-1" in citations
    assert not any("9999.3" in c or "comment 3" in c for c in citations)


def test_build_eval_rejects_a_section_with_no_chunks(corpus, tmp_path):
    bad = [{"_id": "q1", "text": "Anything?", "section": "9999.99"}]
    with pytest.raises(ValueError, match="9999.99"):
        build_eval(corpus, tmp_path / "eval", bad)


@pytest.fixture
def index(corpus):
    idx = PolicyIndex(open_client(), HashingEncoder(512))
    idx.build(load_corpus(corpus))
    return idx


class CiteTop:
    name = "cite-top"

    def draft(self, question, passages, feedback=""):
        return Draft(True, (Claim("Something.", (passages[0].chunk_id,)),))


def test_evaluate_agent_scores_answers_and_refusals(index):
    result = evaluate_agent(PolicyAgent(index, CiteTop(), k=3), QUESTIONS)
    rows = {r["_id"]: r for r in result["questions"]}
    assert rows["q1"]["status"] == "answered" and rows["q1"]["correct"]
    assert rows["oos1"]["status"] == "answered" and not rows["oos1"]["correct"]
    s = result["summary"]
    assert (s["in_scope"], s["out_of_scope"]) == (2, 1)
    assert s["answered_in_scope"] == 1.0
    assert s["refused_out_of_scope"] == 0.0

    extractive = evaluate_agent(PolicyAgent(index, ExtractiveAnswerer(), k=3), QUESTIONS)
    assert extractive["summary"]["refused_out_of_scope"] == 1.0


class Flaky:
    """Cites the top passage, but fails on chosen questions."""

    name = "flaky"

    def __init__(self, fail):
        self.fail = fail  # question text -> the exception to raise
        self.asked = []

    def draft(self, question, passages, feedback=""):
        self.asked.append(question)
        if question in self.fail:
            raise self.fail[question]
        return Draft(True, (Claim("Something.", (passages[0].chunk_id,)),))


class Stop(Exception):
    pass


def test_evaluate_agent_stops_and_resumes(index):
    q2 = QUESTIONS[1]["text"]
    answerer = Flaky({q2: Stop("rate limited")})
    saved = []
    first = evaluate_agent(
        PolicyAgent(index, answerer, k=3),
        QUESTIONS,
        on_row=saved.append,
        errors=(Stop,),
        halt=lambda err: True,
    )
    assert not first["complete"] and first["stopped"] == "rate limited"
    assert (first["scored"], first["total"]) == (1, 3)
    assert [r["_id"] for r in first["questions"]] == ["q1"]
    assert len(saved) == 1

    answerer = Flaky({})
    second = evaluate_agent(PolicyAgent(index, answerer, k=3), QUESTIONS, done=first["questions"])
    assert second["complete"] and second["stopped"] is None
    assert [r["_id"] for r in second["questions"]] == ["q1", "q2", "oos1"]
    assert second["questions"][0] is first["questions"][0]
    assert QUESTIONS[0]["text"] not in answerer.asked


def test_evaluate_agent_records_errors_and_keeps_to_a_budget(index):
    q1 = QUESTIONS[0]["text"]
    answerer = Flaky({q1: ValueError("no JSON object in the reply")})
    result = evaluate_agent(
        PolicyAgent(index, answerer, k=3), QUESTIONS, errors=(ValueError,), halt=lambda e: False
    )
    row = result["questions"][0]
    assert row["status"] == "error" and not row["correct"]
    assert "no JSON object" in row["reason"]
    assert result["summary"]["errors"] == 1 and result["complete"]

    asked = iter([True, False])
    result = evaluate_agent(
        PolicyAgent(index, Flaky({}), k=3), QUESTIONS, can_ask=lambda: next(asked)
    )
    assert result["scored"] == 1 and result["stopped"] == "the request budget is spent"

    with pytest.raises(ValueError):  # errors not listed are not swallowed
        evaluate_agent(PolicyAgent(index, Flaky({q1: ValueError("x")}), k=3), QUESTIONS)


def test_summarize_handles_no_answers():
    rows = [
        {"section": "1.1", "status": "refused", "correct": False, "retrieved_relevant": True},
        {"section": None, "status": "refused", "correct": True, "retrieved_relevant": False},
    ]
    s = summarize(rows)
    assert s["answered_in_scope"] == 0.0
    assert s["correct_when_answered"] is None
    assert s["relevant_retrieved"] == 1.0
    assert s["refused_out_of_scope"] == 1.0
    table = markdown_summary([{"answerer": "x", "k": 5, "summary": s}])
    assert "| x | 5 | 0% | 0% | - | 100% |" in table
    partial = {"answerer": "x", "k": 5, "summary": s, "complete": False, "scored": 2}
    assert "| x (partial, 2 of 61) |" in markdown_summary([{**partial, "total": 61}])


def test_eval_commands(corpus, tmp_path, capsys):
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    qdrant = str(tmp_path / "qdrant")
    cli.main(
        ["index", "build", "--corpus", str(corpus), "--qdrant", qdrant, "--model", "hashing-512"]
    )
    capsys.readouterr()
    ev = ["eval", "build", "--corpus", str(corpus), "--out", str(tmp_path / "eval")]
    assert cli.main([*ev, "--questions", str(qfile)]) == 0
    assert json.loads(capsys.readouterr().out)["queries"] == 2
    out = tmp_path / "results" / "agent.json"
    code = cli.main(
        [
            "eval",
            "agent",
            "--questions",
            str(qfile),
            "--qdrant",
            qdrant,
            "-k",
            "3",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["answerer"] == "extractive" and saved["k"] == 3
    assert saved["encoder"] == "hashing-512" and saved["chunks"] > 0
    assert len(saved["questions"]) == 3
    capsys.readouterr()
    cli.main(["eval", "report", str(out)])
    assert "| extractive | 3 |" in capsys.readouterr().out


def test_eval_agent_resumes(corpus, tmp_path, capsys, monkeypatch):
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    qdrant = str(tmp_path / "qdrant")
    cli.main(
        ["index", "build", "--corpus", str(corpus), "--qdrant", qdrant, "--model", "hashing-512"]
    )
    out = tmp_path / "agent.json"
    base = ["eval", "agent", "--questions", str(qfile), "--qdrant", qdrant, "--out", str(out)]
    assert cli.main([*base, "-k", "3", "--limit", "1"]) == 0
    first = json.loads(out.read_text())
    assert first["scored"] == 1 and first["complete"]  # complete for the questions it was given
    assert first["corpus_date"] == "2026-01-01"

    assert cli.main([*base, "-k", "3", "--resume"]) == 0
    second = json.loads(out.read_text())
    assert second["complete"] and second["scored"] == 3
    assert second["questions"][0] == first["questions"][0]
    assert "resuming: 1 questions already scored" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="different k"):
        cli.main([*base, "-k", "2", "--resume"])
    with pytest.raises(SystemExit, match="does not count requests"):
        cli.main([*base, "-k", "3", "--max-requests", "5"])


def test_eval_agent_keeps_to_max_requests(corpus, tmp_path, capsys, monkeypatch):
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    qdrant = str(tmp_path / "qdrant")
    cli.main(
        ["index", "build", "--corpus", str(corpus), "--qdrant", qdrant, "--model", "hashing-512"]
    )

    class Counting(Flaky):
        name = "openrouter:fake"

        def __init__(self):
            super().__init__({})
            self.requests = 0

        def draft(self, question, passages, feedback=""):
            self.requests += 1
            return super().draft(question, passages, feedback)

    import policy_assistant.answerers as answerers

    monkeypatch.setattr(answerers, "make_answerer", lambda name, **kw: Counting())
    out = tmp_path / "agent.json"
    args = ["eval", "agent", "--questions", str(qfile), "--qdrant", qdrant, "--out", str(out)]
    # A question may take two requests (one retry), so with 3 allowed a third question,
    # asked after 2 requests, might not fit and is left for later.
    assert cli.main([*args, "-k", "3", "--max-requests", "3"]) == 0
    saved = json.loads(out.read_text())
    assert saved["answerer"] == "openrouter:fake"
    assert (saved["scored"], saved["complete"]) == (2, False)
    assert saved["stopped"] == "the request budget is spent"
    err = capsys.readouterr().err
    assert "scored 2 of 3 questions, 2 requests sent" in err and "--resume" in err
