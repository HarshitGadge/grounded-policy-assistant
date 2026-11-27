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
