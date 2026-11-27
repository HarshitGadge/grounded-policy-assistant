import gzip
import json
import urllib.error

import pytest

from policy_assistant import ecfr


def sections(part_xml):
    return {s.number: s for s in ecfr.parse_part(part_xml, 99, "9999")}


def test_parses_sections_and_appendices(part_xml):
    found = sections(part_xml)
    assert list(found) == [
        "9999.1",
        "9999.2",
        "9999.3",
        "Appendix A to Part 9999",
        "Supplement I to Part 9999",
    ]  # the [Reserved] section has no text and is dropped
    assert found["9999.2"].heading == "Definitions."
    assert found["9999.2"].kind == "section"
    assert found["Appendix A to Part 9999"].kind == "appendix"


def test_paragraph_labels(part_xml):
    paras = sections(part_xml)["9999.2"].paragraphs
    by_text = {" ".join(p.text.split(" ")[:2]): p.label for p in paras}
    assert paras[0].label == ()  # the intro sentence before (a)
    assert by_text["Access device"] == ("a", "1")
    assert by_text["It includes"] == ("b", "1", "i")
    assert by_text["It does"] == ("b", "1", "ii")
    # unmarked text continues the paragraph above it
    assert by_text["This paragraph"] == ("b", "1", "ii")
    assert by_text["Preauthorized transfer"] == ("i",)  # (i) after (h)(1), followed by (j)
    assert by_text["Unauthorized transfer"] == ("k",)  # text inside EXTRACT is still read


def test_notes_and_citations_are_skipped(part_xml):
    text = " ".join(p.text for s in sections(part_xml).values() for p in s.paragraphs)
    assert "footnote" not in text
    assert "00 FR" not in text
    assert "Authority:" not in text


def test_appendix_subheads(part_xml):
    supp = sections(part_xml)["Supplement I to Part 9999"]
    assert [p.subhead for p in supp.paragraphs] == ["2(a) Access Device"] * 2 + ["2(b)(1) Account"]
    assert all(p.label == () for p in supp.paragraphs)


def test_part_url():
    assert ecfr.part_url(12, "1005", "2026-01-01") == (
        "https://www.ecfr.gov/api/versioner/v1/full/2026-01-01/title-12.xml?part=1005"
    )


class FakeResponse:
    def __init__(self, body: bytes, headers: dict | None = None):
        self.body = body
        self.headers = headers or {}

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_latest_date(monkeypatch):
    body = json.dumps({"titles": [{"number": 12, "up_to_date_as_of": "2026-11-20"}]}).encode()
    monkeypatch.setattr(ecfr.urllib.request, "urlopen", lambda req, timeout: FakeResponse(body))
    assert ecfr.latest_date(12) == "2026-11-20"
    with pytest.raises(ValueError):
        ecfr.latest_date(7)


def test_fetch_retries_when_busy(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "busy", {}, None)
        return FakeResponse(b"<DIV5/>")

    monkeypatch.setattr(ecfr.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ecfr.time, "sleep", lambda s: None)
    assert ecfr.fetch_part(12, "1005", "2026-01-01") == b"<DIV5/>"
    assert len(calls) == 3
    assert "grounded-policy-assistant" in ecfr.USER_AGENT


def test_fetch_asks_for_xml(monkeypatch):
    seen = {}

    def urlopen(req, timeout):
        seen.update({k.lower(): v for k, v in req.header_items()})
        return FakeResponse(b"<DIV5/>")

    monkeypatch.setattr(ecfr.urllib.request, "urlopen", urlopen)
    ecfr.fetch_part(12, "1005", "2026-01-01")
    assert seen["accept"] == "application/xml"
    assert seen["accept-encoding"] == "gzip"  # eCFR refuses uncompressed full text


def test_fetch_decompresses_gzip(monkeypatch):
    body = gzip.compress(b"<DIV5/>")
    monkeypatch.setattr(
        ecfr.urllib.request,
        "urlopen",
        lambda req, timeout: FakeResponse(body, {"Content-Encoding": "gzip"}),
    )
    assert ecfr.fetch_part(12, "1005", "2026-01-01") == b"<DIV5/>"


def test_fetch_does_not_retry_not_found(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(ecfr.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="eCFR answered 404"):
        ecfr.fetch_part(12, "1005", "2026-01-01")
    assert len(calls) == 1
