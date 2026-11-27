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


def _comment_citation(subhead: str, group: list[Paragraph]) -> str | None:
    key = _COMMENT_KEY.match(subhead)
    if not key:
        return None
    numbers = [m.group(1) for p in group if (m := _COMMENT_NUMBER.match(p.text))]
    if not numbers:
        return f"comment {key.group(1)}"
    if numbers[0] == numbers[-1]:
        return f"comment {key.group(1)}-{numbers[0]}"
    return f"comments {key.group(1)}-{numbers[0]} through {key.group(1)}-{numbers[-1]}"


def _citation(section: Section, group: list[Paragraph]) -> str:
    if section.kind == "appendix":
        base = f"{section.title} CFR part {section.part}, {section.number}"
        subhead = group[0].subhead
        if not subhead:
            return base
        if section.number.lower().startswith("supplement i "):
            comment = _comment_citation(subhead, group)
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
    for group in _groups(section, max_chars):
        labels = [p.label for p in group]
        citation = _citation(section, group)
        header = f"{citation} {section.heading}".strip()
        body = "\n".join(_line(section, p) for p in group)
        text = f"{header}\n{body}"
        digest = hashlib.sha256(f"{date}\n{citation}\n{text}".encode()).hexdigest()[:16]
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
