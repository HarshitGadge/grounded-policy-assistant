import pytest

from policy_assistant import cli, corpus


@pytest.fixture
def offline(monkeypatch, part_xml):
    monkeypatch.setattr(corpus, "fetch_part", lambda t, p, d: part_xml)
    monkeypatch.setattr(cli, "latest_date", lambda title: "2026-11-20")


def test_build_and_stats(tmp_path, offline, capsys):
    out = tmp_path / "corpus"
    assert (
        cli.main(
            ["corpus", "build", "--date", "2026-01-01", "--part", "99:9999", "--out", str(out)]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "99 CFR part 9999: 3 sections, 2 appendices" in printed
    assert (out / "corpus.jsonl").exists()
    assert cli.main(["corpus", "stats", str(out), "--show", "1"]) == 0
    printed = capsys.readouterr().out
    assert "99 CFR 9999" in printed
    assert "characters per chunk" in printed


def test_latest_date(tmp_path, offline, capsys):
    out = tmp_path / "corpus"
    cli.main(["corpus", "build", "--date", "latest", "--part", "99:9999", "--out", str(out)])
    assert "2026-11-20" in capsys.readouterr().out
    assert (out / "raw" / "title-99-part-9999-2026-11-20.xml").exists()


def test_bad_date(tmp_path, offline):
    with pytest.raises(SystemExit):
        cli.main(["corpus", "build", "--date", "Jan 1", "--out", str(tmp_path)])


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_index_build_and_search(tmp_path, offline, capsys):
    out = tmp_path / "corpus"
    qdrant = str(tmp_path / "qdrant")
    cli.main(["corpus", "build", "--date", "2026-01-01", "--part", "99:9999", "--out", str(out)])
    capsys.readouterr()
    args = ["--corpus", str(out), "--qdrant", qdrant, "--model", "hashing-512"]
    assert cli.main(["index", "build", *args]) == 0
    assert "with hashing-512 (512 dims)" in capsys.readouterr().out
    # search reads the encoder from the collection, so --model is not needed.
    assert (
        cli.main(["search", "What is a preauthorized transfer?", "--qdrant", qdrant, "-k", "2"])
        == 0
    )
    printed = capsys.readouterr().out
    assert printed.startswith("1. [")
    assert "2. [" in printed
    assert "[99 CFR 9999.2(i)]" in printed
    assert cli.main(["search", "transfer", "--qdrant", qdrant, "--part", "1005"]) == 0
    assert capsys.readouterr().out == ""


def test_search_without_an_index(tmp_path):
    with pytest.raises(SystemExit, match="run policy-assistant index build"):
        cli.main(["search", "anything", "--qdrant", str(tmp_path / "none")])
