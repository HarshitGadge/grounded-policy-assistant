import json

import pytest

from policy_assistant.corpus import build_corpus, load_corpus, parse_part_spec


def fake_fetch(part_xml):
    calls = []

    def fetch(title, part, date):
        calls.append((title, part, date))
        return part_xml

    return fetch, calls


def test_build_writes_beir_corpus_and_manifest(tmp_path, part_xml):
    fetch, calls = fake_fetch(part_xml)
    manifest = build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=fetch)
    records = load_corpus(tmp_path)
    assert calls == [(99, "9999", "2026-01-01")]
    assert manifest["chunks"] == len(records) > 0
    assert manifest["parts"][0]["sections"] == 3
    assert manifest["parts"][0]["appendices"] == 2
    assert len(manifest["parts"][0]["sha256"]) == 64
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    first = records[0]
    assert set(first) >= {"_id", "title", "text", "citation", "section", "date"}
    assert first["title"].startswith(first["citation"])
    assert len({r["_id"] for r in records}) == len(records)


def test_rebuild_reads_the_saved_xml(tmp_path, part_xml):
    fetch, calls = fake_fetch(part_xml)
    first = build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=fetch)
    second = build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=fetch)
    assert len(calls) == 1
    assert first == second


def test_a_section_listed_twice_is_an_error(tmp_path, part_xml):
    twice = part_xml.replace(
        b"</DIV6>",
        b'<DIV8 N="9999.1" TYPE="SECTION"><HEAD>\xc2\xa7 9999.1 Again.</HEAD><P>(a) Text.</P></DIV8></DIV6>',
    )
    with pytest.raises(ValueError, match="more than once"):
        build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=lambda *a: twice)


def test_empty_part_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no sections"):
        build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=lambda *a: b"<DIV5/>")


def test_ragbench_can_load_the_corpus(tmp_path, part_xml):
    # RAGBench reads BEIR folders; a corpus plus one query and judgement must load.
    ragbench = pytest.importorskip("ragbench")
    fetch, _ = fake_fetch(part_xml)
    build_corpus(tmp_path, "2026-01-01", [(99, "9999")], fetch=fetch)
    doc = load_corpus(tmp_path)[0]["_id"]
    (tmp_path / "queries.jsonl").write_text('{"_id": "q1", "text": "what is an account"}\n')
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(f"query-id\tcorpus-id\tscore\nq1\t{doc}\t1\n")
    data = ragbench.load_beir(tmp_path)
    assert data.summary()["documents"] == len(load_corpus(tmp_path))


@pytest.mark.parametrize(
    "spec, expected",
    [("12:1005", (12, "1005")), ("12 CFR 1005", (12, "1005")), (" 12 : 1016 ", (12, "1016"))],
)
def test_parse_part_spec(spec, expected):
    assert parse_part_spec(spec) == expected


@pytest.mark.parametrize("spec", ["1005", "12:1005.2", "twelve:1005"])
def test_parse_part_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_part_spec(spec)
