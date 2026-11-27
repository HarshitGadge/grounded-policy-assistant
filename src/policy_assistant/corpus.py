"""Build the regulation corpus: download eCFR parts, parse them, chunk them, write them out.

The output folder holds:

- ``corpus.jsonl``: one chunk per line, in the BEIR corpus layout (``_id``, ``title``,
  ``text``) with the chunk's citation and position as extra fields, so RAGBench and any
  BEIR tool can read it as is.
- ``manifest.json``: the eCFR date, the parts, each part's source URL and SHA-256, and
  the counts, so a corpus can be rebuilt and checked.
- ``raw/``: the downloaded XML. A rebuild reads it from here instead of downloading again.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .chunk import DEFAULT_MAX_CHARS, Chunk, chunk_section
from .ecfr import fetch_part, parse_part, part_url

# Consumer-finance regulations from the CFPB: plain-language rules people ask about,
# with official interpretations that the answers can cite.
DEFAULT_PARTS: tuple[tuple[int, str], ...] = (
    (12, "1002"),  # Regulation B, Equal Credit Opportunity Act
    (12, "1005"),  # Regulation E, Electronic Fund Transfers
    (12, "1016"),  # Regulation P, Privacy of Consumer Financial Information
    (12, "1030"),  # Regulation DD, Truth in Savings
)

Fetch = Callable[[int, str, str], bytes]


@dataclass
class PartStats:
    title: int
    part: str
    url: str
    sha256: str
    sections: int
    appendices: int
    chunks: int


def parse_part_spec(spec: str) -> tuple[int, str]:
    """Turn "12:1005" or "12 CFR 1005" into (12, "1005")."""
    cleaned = spec.replace("CFR", " ").replace(":", " ").split()
    if len(cleaned) != 2 or not cleaned[0].isdigit() or not cleaned[1].isdigit():
        raise ValueError(f"expected TITLE:PART such as 12:1005, got {spec!r}")
    return int(cleaned[0]), cleaned[1]


def _raw_xml(title: int, part: str, date: str, raw_dir: Path, fetch: Fetch) -> bytes:
    path = raw_dir / f"title-{title}-part-{part}-{date}.xml"
    if path.exists():
        return path.read_bytes()
    xml = fetch(title, part, date)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xml)
    return xml


def build_chunks(
    parts: Iterable[tuple[int, str]],
    date: str,
    raw_dir: Path,
    fetch: Fetch = fetch_part,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[Chunk], list[PartStats]]:
    chunks: list[Chunk] = []
    stats: list[PartStats] = []
    for title, part in parts:
        xml = _raw_xml(title, part, date, raw_dir, fetch)
        sections = parse_part(xml, title, part)
        numbers = Counter(s.number for s in sections)
        repeated = sorted(n for n, k in numbers.items() if k > 1)
        if repeated:
            raise ValueError(f"{title} CFR part {part} lists these more than once: {repeated[:10]}")
        if not sections:
            raise ValueError(f"no sections found in {title} CFR part {part} on {date}")
        part_chunks = [c for s in sections for c in chunk_section(s, date, max_chars)]
        chunks.extend(part_chunks)
        stats.append(
            PartStats(
                title=title,
                part=part,
                url=part_url(title, part, date),
                sha256=hashlib.sha256(xml).hexdigest(),
                sections=sum(s.kind == "section" for s in sections),
                appendices=sum(s.kind == "appendix" for s in sections),
                chunks=len(part_chunks),
            )
        )
    seen: dict[str, Chunk] = {}
    for c in chunks:
        if c.id in seen:
            raise ValueError(f"duplicate chunk id {c.id}: {seen[c.id].citation} and {c.citation}")
        seen[c.id] = c
    return chunks, stats


def beir_record(chunk: Chunk) -> dict:
    """A corpus.jsonl line: BEIR's _id/title/text plus where the text comes from."""
    title, _, body = chunk.text.partition("\n")
    return {
        "_id": chunk.id,
        "title": title,
        "text": body,
        "citation": chunk.citation,
        "cfr_title": chunk.title,
        "part": chunk.part,
        "section": chunk.section,
        "heading": chunk.heading,
        "paragraphs": chunk.labels,
        "date": chunk.date,
    }


def build_corpus(
    out_dir: str | Path,
    date: str,
    parts: Iterable[tuple[int, str]] = DEFAULT_PARTS,
    fetch: Fetch | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    """Build the corpus into `out_dir` and return its manifest.

    `fetch(title, part, date)` returns a part's XML; it defaults to downloading from eCFR.
    """
    fetch = fetch or fetch_part
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunks, stats = build_chunks(list(parts), date, out / "raw", fetch, max_chars)
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(beir_record(chunk), ensure_ascii=False) + "\n")
    manifest = {
        "source": "eCFR (https://www.ecfr.gov), U.S. government work in the public domain",
        "date": date,
        "max_chars": max_chars,
        "builder": f"grounded-policy-assistant {__version__}",
        "parts": [vars(s) for s in stats],
        "chunks": len(chunks),
        "characters": sum(len(c.text) for c in chunks),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_corpus(folder: str | Path) -> list[dict]:
    with (Path(folder) / "corpus.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
