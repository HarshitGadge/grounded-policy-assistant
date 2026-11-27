"""Fetch and parse regulation text from the electronic Code of Federal Regulations.

eCFR (https://www.ecfr.gov) publishes the current and past text of every CFR
part through a public API, and the text is a U.S. government work in the
public domain. A request is pinned to a date, so a corpus built for the same
date and parts is the same text every time.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .paragraphs import LabelTracker, leading_markers

API = "https://www.ecfr.gov/api/versioner/v1"
USER_AGENT = (
    "grounded-policy-assistant (+https://github.com/HarshitGadge/grounded-policy-assistant)"
)

# Elements that hold a paragraph of text inside a section or appendix:
# P, and flush paragraphs FP, FP-1, FP-2 and so on.
_TEXT = re.compile(r"^(P|FP(-\d+)?)$")
# Headings inside an appendix. eCFR writes them as HD1/HD2/HD3; some parts use HED or HD.
_SUBHEAD = re.compile(r"^(HD\d*|HED)$")
# Elements whose text is not regulation text: source notes, editorial notes, footnotes.
_SKIP_TAGS = {"CITA", "EDNOTE", "FTNT", "SECAUTH", "AUTH", "SOURCE", "HEAD"}
_SECTION_NUMBER = re.compile(r"(\d+\.\d+[a-z]?)")
_SPACE = re.compile(r"\s+")


@dataclass
class Paragraph:
    label: tuple[str, ...]
    text: str
    subhead: str = ""  # the closest heading above it inside an appendix, e.g. "2(a) Access device"


@dataclass
class Section:
    title: int
    part: str
    number: str  # "1005.2" for a section, "Appendix A to Part 1005" for an appendix
    kind: str  # "section" or "appendix"
    heading: str
    paragraphs: list[Paragraph] = field(default_factory=list)


def part_url(title: int, part: str, date: str) -> str:
    return f"{API}/full/{date}/title-{title}.xml?part={part}"


def _get(url: str, timeout: float, accept: str, attempts: int = 4) -> bytes:
    # eCFR's full-text endpoint answers 406 Not Acceptable unless the request
    # names the format and allows a compressed response.
    headers = {"User-Agent": USER_AGENT, "Accept": accept, "Accept-Encoding": "gzip"}
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                if (response.headers.get("Content-Encoding") or "").lower() == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as err:
            # eCFR answers 429 when it is busy; anything else is a real error.
            if (err.code != 429 and err.code < 500) or attempt == attempts - 1:
                detail = err.read()[:500].decode("utf-8", "replace") if err.fp else ""
                raise RuntimeError(f"eCFR answered {err.code} for {url}: {detail}") from err
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_part(title: int, part: str, date: str, timeout: float = 120) -> bytes:
    """Download one part as eCFR XML, as it read on `date` (YYYY-MM-DD)."""
    return _get(part_url(title, part, date), timeout, "application/xml")


def latest_date(title: int, timeout: float = 60) -> str:
    """The newest date eCFR has text for in `title`."""
    titles = json.loads(_get(f"{API}/titles.json", timeout, "application/json"))["titles"]
    for t in titles:
        if t["number"] == title:
            return t["up_to_date_as_of"]
    raise ValueError(f"eCFR has no title {title}")


def _text(el: ET.Element) -> str:
    return _SPACE.sub(" ", "".join(el.itertext())).strip()


def _paragraph_elements(div: ET.Element):
    """Yield ("p", element) and ("hd", element) in document order, skipping notes."""
    for child in div:
        if child.tag in _SKIP_TAGS:
            continue
        if _TEXT.match(child.tag):
            yield "p", child
        elif _SUBHEAD.match(child.tag):
            yield "hd", child
        elif child.get("TYPE") in {"SECTION", "APPENDIX"}:
            continue  # parsed as its own section
        else:
            yield from _paragraph_elements(child)


def _section_number(n: str, head: str) -> str | None:
    for source in (n, head):
        m = _SECTION_NUMBER.search(source or "")
        if m:
            return m.group(1)
    return None


def _heading(head: str, number: str) -> str:
    # "§ 1005.2 Definitions." -> "Definitions."
    return re.sub(rf"^§+\s*{re.escape(number)}\s*", "", head).strip()


# "(g) Error resolution standards—(1) Compliance program. ..." or
# "(5) Notice to financial institution. (i) Notice is given ...": eCFR prints a
# heading and the first child paragraph in one element.
_INLINE_CHILD = re.compile(r"^([^()]{1,200}?(?:—|\.))\s*\(([0-9a-zA-Z]+)\)\s+(?=\S)")

# Headings inside official interpretations (Supplement I). They name the rule
# paragraph that the comments under them explain.
_SUPPLEMENT_SECTION = re.compile(r"^Section\s+\d+\.(\d+[a-z]?)\b")
_COMMENT_HEADING = re.compile(r"^(?:Paragraphs?\s+)?(\d+[a-z]?(?:\([0-9a-zA-Z]+\))+)(?:\s+(.*))?$")
_MARKERS_HEADING = re.compile(r"^((?:\([0-9a-zA-Z]+\))+)\s+([A-Z].*)$")
_MAX_HEADING = 150
# "7(a) Individual accounts. 1. Open-end credit ..." in one element.
_HEADING_THEN_COMMENT = re.compile(r"^(.{1,150}?\.)\s+(\d+\.\s.+)$")


def _comment_heading(text: str, section_key: str | None) -> str | None:
    """ "Paragraph 2(b)(2)" -> "2(b)(2)"; "(a)(1) Account opening." under section 4 ->
    "4(a)(1) Account opening."; anything else -> None."""
    if len(text) > _MAX_HEADING:
        return None
    m = _COMMENT_HEADING.match(text)
    if m:
        return f"{m.group(1)} {m.group(2) or ''}".strip()
    m = _MARKERS_HEADING.match(text)
    if m and section_key:
        return f"{section_key}{m.group(1)} {m.group(2)}"
    return None


def _items(div: ET.Element) -> list[tuple[str, str]]:
    items = [(kind, _text(el)) for kind, el in _paragraph_elements(div)]
    return [(kind, t) for kind, t in items if t]


def _parse_appendix(section: Section, div: ET.Element) -> None:
    supplement = section.number.lower().startswith("supplement i ")
    subhead, section_key = "", None
    for kind, text in _items(div):
        if supplement:
            m = _SUPPLEMENT_SECTION.match(text)
            if m:
                subhead, section_key = text, m.group(1)
                continue
            joined = _HEADING_THEN_COMMENT.match(text)
            if joined and (heading := _comment_heading(joined.group(1), section_key)):
                subhead, text = heading, joined.group(2)
            elif heading := _comment_heading(text, section_key):
                subhead = heading
                continue
        if kind == "hd":
            subhead = text
        else:
            section.paragraphs.append(Paragraph((), text, subhead))


def _parse_section(section: Section, div: ET.Element) -> None:
    elements = [t for kind, t in _items(div) if kind == "p"]
    parsed = [leading_markers(t) for t in elements]
    tracker = LabelTracker()
    label: tuple[str, ...] = ()

    def next_marker(i: int) -> str | None:
        return next((m[0] for m, _ in parsed[i + 1 :] if m), None)

    for i, (markers, rest) in enumerate(parsed):
        if not markers:
            # An unmarked paragraph continues the one above it.
            section.paragraphs.append(Paragraph(label, elements[i]))
            continue
        for j, marker in enumerate(markers):
            after = markers[j + 1] if j + 1 < len(markers) else next_marker(i)
            try:
                label = tracker.push(marker, after)
            except ValueError:
                break
        inline = _INLINE_CHILD.match(rest)
        if inline:
            remainder = rest[inline.end() :]
            trial = tracker.copy()
            child = inline.group(2)
            after = (leading_markers(remainder)[0] or [None])[0] or next_marker(i)
            try:
                child_label = trial.push(child, after)
            except ValueError:
                child_label = ()
            if child_label == (*label, child):
                section.paragraphs.append(Paragraph(label, inline.group(1)))
                tracker, label, rest = trial, child_label, remainder
        if rest:
            section.paragraphs.append(Paragraph(label, rest))


def parse_part(xml: bytes, title: int, part: str) -> list[Section]:
    """Parse eCFR XML for one part into sections with labelled paragraphs."""
    root = ET.fromstring(xml)
    sections: list[Section] = []
    for div in root.iter():
        kind = (div.get("TYPE") or "").upper()
        if kind not in {"SECTION", "APPENDIX"}:
            continue
        head_el = div.find("HEAD")
        head = _text(head_el) if head_el is not None else ""
        if kind == "SECTION":
            number = _section_number(div.get("N", ""), head)
            if number is None or not number.startswith(f"{part}."):
                continue
            section = Section(title, part, number, "section", _heading(head, number))
            _parse_section(section, div)
        else:
            section = Section(title, part, div.get("N") or head, "appendix", head)
            _parse_appendix(section, div)
        if section.paragraphs:
            sections.append(section)
    return sections
