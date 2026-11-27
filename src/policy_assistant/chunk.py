"""Split regulation sections into citable chunks for retrieval.

Chunks follow the regulation's own structure: a chunk never mixes two
top-level paragraphs such as (a) and (b), and it is cited by the deepest
paragraph that contains all of its text. Each chunk starts with its
citation and section heading, so a retrieved chunk says where it is from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .ecfr import Paragraph, Section
from .paragraphs import format_citation

DEFAULT_MAX_CHARS = 1500


@dataclass
class Chunk:
    id: str
    title: int
    part: str
    section: str
    heading: str
    citation: str
    labels: list[str]
    text: str
    date: str

    def to_dict(self) -> dict:
        return asdict(self)


def _common_prefix(labels: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not labels:
        return ()
    prefix = labels[0]
    for label in labels[1:]:
        n = 0
        while n < min(len(prefix), len(label)) and prefix[n] == label[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


# A heading in an official interpretation, such as "2(a) Access Device" or
# "6(b)(3) Timing", names the paragraph of the rule its comments explain.
_COMMENT_KEY = re.compile(r"^(\d+[a-z]?(?:\([0-9a-zA-Z]+\))*)(?=\s|$)")
_COMMENT_NUMBER = re.compile(r"^(\d+)\.\s")


def _comment_numbers(section: Section) -> list[str | None]:
    """The comment number each paragraph of an interpretation belongs to.

    Comments are numbered 1., 2., ... under each heading, and their sub-points
    (i., ii.) belong to the comment above them.
    """
    numbers: list[str | None] = []
    current, subhead = None, None
    for p in section.paragraphs:
        if p.subhead != subhead:
            current, subhead = None, p.subhead
        m = _COMMENT_NUMBER.match(p.text)
        if m:
            current = m.group(1)
        numbers.append(current)
    return numbers


def _comment_citation(subhead: str, numbers: list[str | None]) -> str | None:
    key = _COMMENT_KEY.match(subhead)
    if not key:
        return None
    k = key.group(1)
    found = [n for n in numbers if n]
    if not found:
        return f"comment {k}"
    if found[0] == found[-1]:
        return f"comment {k}-{found[0]}"
    return f"comments {k}-{found[0]} through {k}-{found[-1]}"


def _citation(section: Section, group: list[Paragraph], numbers: list[str | None]) -> str:
    if section.kind == "appendix":
        base = f"{section.title} CFR part {section.part}, {section.number}"
        subhead = group[0].subhead
        if not subhead:
            return base
        if section.number.lower().startswith("supplement i "):
            comment = _comment_citation(subhead, numbers)
            if comment:
                return f"{base}, {comment}"
        return f"{base}, {subhead}"
    return format_citation(section.title, section.number, _common_prefix([p.label for p in group]))


def _line(section: Section, p: Paragraph) -> str:
    """A paragraph as it appears in a chunk: "(b)(1)(ii) Text" for a section."""
    if section.kind == "section" and p.label:
        return "".join(f"({m})" for m in p.label) + " " + p.text
    return p.text


def _groups(section: Section, max_chars: int) -> list[list[Paragraph]]:
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    size = 0
    for p in section.paragraphs:
        line = len(_line(section, p))
        top = (p.label[:1], p.subhead)
        new_top = bool(current) and (current[-1].label[:1], current[-1].subhead) != top
        too_big = bool(current) and size + 1 + line > max_chars
        if new_top or too_big:
            groups.append(current)
            current, size = [], 0
        current.append(p)
        size += line + (1 if size else 0)
    if current:
        groups.append(current)
    return groups


def chunk_section(section: Section, date: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    chunks = []
    numbers = _comment_numbers(section)
    index = {id(p): i for i, p in enumerate(section.paragraphs)}
    for position, group in enumerate(_groups(section, max_chars)):
        labels = [p.label for p in group]
        citation = _citation(section, group, [numbers[index[id(p)]] for p in group])
        header = f"{citation} {section.heading}".strip()
        lines = [_line(section, p) for p in group]
        if section.kind == "appendix" and group[0].subhead:
            lines.insert(0, group[0].subhead)  # keeps the heading's words with the text
        body = "\n".join(lines)
        text = f"{header}\n{body}"
        # The position keeps two chunks apart when a regulation repeats the same
        # wording under one heading, as model forms do.
        key = f"{date}\n{section.title}\n{section.part}\n{section.number}\n{position}\n{text}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=digest,
                title=section.title,
                part=section.part,
                section=section.number,
                heading=section.heading,
                citation=citation,
                labels=sorted({"".join(f"({m})" for m in label) for label in labels if label}),
                text=text,
                date=date,
            )
        )
    return chunks
