import pytest

from policy_assistant.corpus import build_corpus, load_corpus
from policy_assistant.embed import HashingEncoder
from policy_assistant.retriever import QdrantRetriever

ragbench = pytest.importorskip("ragbench")


@pytest.fixture
def dataset(tmp_path, part_xml):
    """The fixture corpus with three hand-written questions, as a BEIR dataset."""
    build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=lambda t, p, d: part_xml)
    by_citation = {r["citation"]: r["_id"] for r in load_corpus(tmp_path)}
    questions = {
        "q1": ("What is a preauthorized transfer?", "99 CFR 9999.2(i)"),
        "q2": ("What does business day mean?", "99 CFR 9999.2(c)"),
        "q3": (
            "Which procedures must an institution keep for error resolution?",
            "99 CFR 9999.3(c)",
        ),
    }
    (tmp_path / "queries.jsonl").write_text(
        "".join(f'{{"_id": "{q}", "text": "{text}"}}\n' for q, (text, _) in questions.items())
    )
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"{q}\t{by_citation[c]}\t1\n" for q, (_, c) in questions.items())
    )
    return ragbench.load_beir(tmp_path)


def test_ragbench_scores_the_qdrant_retriever(dataset):
    retriever = QdrantRetriever(encoder=HashingEncoder(512))
    result = ragbench.evaluate(retriever, dataset, depth=10, recall_ks=(10,), bootstrap_rounds=50)
    assert result.mean("recall@10") == 1.0
    assert result.mean("mrr@10") > 0.5


def test_search_returns_doc_ids_with_scores(dataset):
    retriever = QdrantRetriever(encoder=HashingEncoder(512))
    retriever.index(dataset.corpus)
    hits = retriever.search("preauthorized transfer", 3)
    assert len(hits) == 3
    assert all(doc_id in dataset.corpus for doc_id, _ in hits)
    assert retriever.search_batch(["preauthorized transfer"], 3) == [hits]


def test_ragbench_can_build_it_from_a_spec():
    from ragbench.retriever import load_retriever

    retriever = load_retriever(
        "policy_assistant.retriever:QdrantRetriever", {"model": "hashing-64"}
    )
    assert retriever.encoder.name == "hashing-64"
