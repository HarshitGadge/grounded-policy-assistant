import pytest

from policy_assistant.corpus import build_corpus, load_corpus
from policy_assistant.embed import HashingEncoder
from policy_assistant.index import (
    PolicyIndex,
    collection_info,
    open_client,
    passage_text,
    point_id,
)


@pytest.fixture
def records(tmp_path, part_xml):
    build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=lambda t, p, d: part_xml)
    return load_corpus(tmp_path)


@pytest.fixture
def index(records):
    idx = PolicyIndex(open_client(), HashingEncoder(512))
    idx.build(records, batch_size=7, metadata={"corpus_date": "2026-01-01"})
    return idx


def top_citations(index, query, k=3, **kwargs):
    return [p.citation for p in index.search(query, k=k, **kwargs)]


def test_build_indexes_every_chunk_once(index, records):
    assert index.count() == len(records)
    # A rebuild replaces the collection; it does not add a second copy.
    index.build(records)
    assert index.count() == len(records)


def test_collection_records_how_it_was_built(index, records):
    info = index.info()
    assert info["encoder"] == "hashing-512"
    assert info["dim"] == 512
    assert info["chunks"] == len(records)
    assert info["corpus_date"] == "2026-01-01"
    assert collection_info(index.client, "missing") is None


def test_search_finds_the_defining_paragraph(index):
    assert top_citations(index, "What does preauthorized transfer mean?")[0] == "99 CFR 9999.2(i)"
    assert top_citations(index, "Who counts as a natural person consumer?", k=2)[0] in {
        "99 CFR 9999.2(d)",
        "99 CFR 9999.2(h)(1)",
    }


def test_search_returns_citable_passages(index, records):
    by_id = {r["_id"]: r for r in records}
    for p in index.search("liability for an unauthorized transfer", k=5):
        source = by_id[p.chunk_id]
        assert p.citation == source["citation"]
        assert p.text == source["text"]
        assert p.part == "9999"
        assert p.date == "2026-01-01"
        assert -1.0 <= p.score <= 1.0


def test_results_are_best_first(index):
    scores = [p.score for p in index.search("two business days notice", k=6)]
    assert scores == sorted(scores, reverse=True)


def test_part_filter(index):
    assert index.search("account", k=3, parts=["9999"])
    assert index.search("account", k=3, parts=["1005"]) == []


def test_search_batch_matches_single_search(index):
    queries = ["business day", "access device"]
    batch = index.search_batch(queries, k=4)
    assert [[p.chunk_id for p in hits] for hits in batch] == [
        [p.chunk_id for p in index.search(q, k=4)] for q in queries
    ]
    assert index.search_batch([], k=4) == []


def test_refuses_a_different_encoder(index):
    other = PolicyIndex(index.client, HashingEncoder(256), index.collection)
    with pytest.raises(ValueError, match="built with hashing-512"):
        other.search("account")


def test_refuses_to_search_a_missing_collection():
    idx = PolicyIndex(open_client(), HashingEncoder(64), "nothing-here")
    with pytest.raises(LookupError, match="build the index first"):
        idx.search("account")


def test_bad_arguments(index, records):
    with pytest.raises(ValueError, match="k must be"):
        index.search("account", k=0)
    with pytest.raises(ValueError, match="batch_size"):
        index.build(records, batch_size=0)
    with pytest.raises(ValueError, match="more than once"):
        index.build(records + records[:1])


def test_point_ids_are_stable_uuids():
    assert point_id("abc") == point_id("abc")
    assert point_id("abc") != point_id("abd")
    assert len(point_id("abc")) == 36


def test_passage_text_puts_the_citation_line_first():
    assert passage_text({"title": "12 CFR 1005.6 Liability", "text": "(a) Text."}) == (
        "12 CFR 1005.6 Liability\n\n(a) Text."
    )
    assert passage_text({"text": " body "}) == "body"


def test_local_folder_index_survives_reopening(tmp_path, records):
    folder = str(tmp_path / "qdrant")
    client = open_client(path=folder)
    PolicyIndex(client, HashingEncoder(128)).build(records)
    client.close()
    reopened = PolicyIndex(open_client(path=folder), HashingEncoder(128))
    assert reopened.count() == len(records)
    assert reopened.search("preauthorized transfer", k=1)[0].citation == "99 CFR 9999.2(i)"
