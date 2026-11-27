"""Fetch and parse regulation text from the electronic Code of Federal Regulations.

eCFR (https://www.ecfr.gov) publishes the current and past text of every CFR
part through a public API, and the text is a U.S. government work in the
public domain. A request is pinned to a date, so a corpus built for the same
date and parts is the same text every time.
"""

from __future__ import annotations

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

# Elements that hold a paragraph of text inside a section or appendix.
_TEXT_TAGS = {"P", "FP"}
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
    # eCFR answers 406 Not Acceptable unless the Accept header names the format asked for.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
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
        if child.tag in _TEXT_TAGS:
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
        else:
            section = Section(title, part, div.get("N") or head, "appendix", head)

        items = [(kind, _text(el)) for kind, el in _paragraph_elements(div)]
        items = [(kind, t) for kind, t in items if t]
        subheads, elements, subhead = [], [], ""
        for kind, t in items:
            if kind == "hd":
                subhead = t
            else:
                subheads.append(subhead)
                elements.append(t)
        tracker = LabelTracker()
        label: tuple[str, ...] = ()
        parsed = [leading_markers(t) for t in elements]
        for i, (markers, rest) in enumerate(parsed):
            if section.kind == "section" and markers:
                for j, marker in enumerate(markers):
                    if j + 1 < len(markers):
                        after = markers[j + 1]
                    else:
                        after = next((m[0] for m, _ in parsed[i + 1 :] if m), None)
                    try:
                        label = tracker.push(marker, after)
                    except ValueError:
                        break
                text = rest
            else:
                # Appendix text and unmarked paragraphs keep the current label:
                # an unmarked paragraph continues the one above it.
                text = elements[i]
            if text:
                section.paragraphs.append(Paragraph(label, text, subheads[i]))
        if section.paragraphs:
            sections.append(section)
    return sections
