"""Command line: ``policy-assistant corpus build`` and ``policy-assistant corpus stats``."""

from __future__ import annotations

import argparse
import re
from collections import Counter

from . import __version__
from .chunk import DEFAULT_MAX_CHARS
from .corpus import DEFAULT_PARTS, build_corpus, load_corpus, parse_part_spec
from .ecfr import latest_date

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _build(args: argparse.Namespace) -> int:
    parts = [parse_part_spec(p) for p in args.part] if args.part else list(DEFAULT_PARTS)
    date = args.date
    if date == "latest":
        titles = {t for t, _ in parts}
        date = min(latest_date(t) for t in titles)
        print(f"latest eCFR date for title(s) {sorted(titles)}: {date}")
    elif not _DATE.match(date):
        raise SystemExit(f"--date must be YYYY-MM-DD or 'latest', got {date!r}")
    manifest = build_corpus(args.out, date, parts, max_chars=args.max_chars)
    for p in manifest["parts"]:
        print(
            f"{p['title']} CFR part {p['part']}: {p['sections']} sections, "
            f"{p['appendices']} appendices, {p['chunks']} chunks"
        )
    print(f"{manifest['chunks']} chunks as of {date} -> {args.out}/corpus.jsonl")
    return 0


def _stats(args: argparse.Namespace) -> int:
    records = load_corpus(args.folder)
    lengths = sorted(len(r["title"]) + 1 + len(r["text"]) for r in records)
    by_part = Counter(f"{r['cfr_title']} CFR {r['part']}" for r in records)
    print(f"{len(records)} chunks")
    for part, n in sorted(by_part.items()):
        print(f"  {part}: {n}")
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(f"characters per chunk: min {lengths[0]}, median {mid}, max {lengths[-1]}")
    for r in records[: args.show]:
        print(f"\n[{r['citation']}]\n{r['title']}\n{r['text'][:300]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-assistant", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="build or inspect the regulation corpus")
    corpus_sub = corpus.add_subparsers(dest="action", required=True)

    build = corpus_sub.add_parser("build", help="download eCFR parts and chunk them")
    build.add_argument("--date", required=True, help="eCFR date, YYYY-MM-DD or 'latest'")
    build.add_argument(
        "--part",
        action="append",
        help="TITLE:PART such as 12:1005; repeat for more (default: Regulations B, E, P, DD)",
    )
    build.add_argument("--out", default="data/corpus", help="output folder")
    build.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    build.set_defaults(func=_build)

    stats = corpus_sub.add_parser("stats", help="summarise a built corpus")
    stats.add_argument("folder", nargs="?", default="data/corpus")
    stats.add_argument("--show", type=int, default=0, help="print the first N chunks")
    stats.set_defaults(func=_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
