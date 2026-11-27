"""Command line for the Grounded Policy Assistant.

* ``policy-assistant corpus build`` / ``corpus stats``: download and chunk regulations.
* ``policy-assistant index build``: embed the corpus into Qdrant.
* ``policy-assistant search``: the closest chunks to a question, with their citations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .chunk import DEFAULT_MAX_CHARS
from .corpus import DEFAULT_PARTS, build_corpus, load_corpus, parse_part_spec
from .ecfr import latest_date
from .embed import DEFAULT_MODEL, make_encoder

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


def _qdrant_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--qdrant", default="data/qdrant", help="local Qdrant folder")
    p.add_argument("--url", help="a Qdrant server instead of the local folder")
    p.add_argument("--collection", default="regulations")


def _open_index(args: argparse.Namespace, model: str):
    from .index import PolicyIndex, open_client

    client = open_client(path=args.qdrant, url=args.url)
    return PolicyIndex(client, make_encoder(model), args.collection, payload_indexes=bool(args.url))


def _index_build(args: argparse.Namespace) -> int:
    records = load_corpus(args.corpus)
    manifest_path = Path(args.corpus) / "manifest.json"
    corpus_date = ""
    if manifest_path.exists():
        corpus_date = json.loads(manifest_path.read_text(encoding="utf-8")).get("date", "")
    index = _open_index(args, args.model)
    try:
        n = index.build(records, batch_size=args.batch_size, metadata={"corpus_date": corpus_date})
    finally:
        index.client.close()
    where = args.url or args.qdrant
    print(f"indexed {n} chunks with {args.model} ({index.encoder.dim} dims) -> {where}")
    return 0


def _open_existing(args: argparse.Namespace):
    """The built index, searched with the encoder that built it (or ``--model``)."""
    from .index import PolicyIndex, collection_info, open_client

    if not args.url and not Path(args.qdrant).exists():
        raise SystemExit(f"no index in {args.qdrant}; run policy-assistant index build first")
    client = open_client(path=args.qdrant, url=args.url)
    info = collection_info(client, args.collection)
    if info is None:
        client.close()
        where = args.url or args.qdrant
        raise SystemExit(f"no collection {args.collection!r} in {where}; run index build first")
    model = getattr(args, "model", None) or info.get("encoder", DEFAULT_MODEL)
    return PolicyIndex(client, make_encoder(model), args.collection)


def _search(args: argparse.Namespace) -> int:
    index = _open_existing(args)
    try:
        passages = index.search(args.question, k=args.k, parts=args.part or None)
    finally:
        index.client.close()
    for rank, p in enumerate(passages, 1):
        print(f"{rank}. [{p.citation}] score {p.score:.3f}")
        print(f"   {p.text[:300].replace(chr(10), ' ')}")
    return 0


def _ask(args: argparse.Namespace) -> int:
    from .agent import PolicyAgent
    from .answerers import make_answerer
    from .openrouter import RateLimited

    options = {"effort": args.effort} if args.answerer.startswith("claude") else {}
    try:
        answerer = make_answerer(args.answerer, **options)
    except ValueError as err:
        raise SystemExit(f"policy-assistant: {err}") from None
    index = _open_existing(args)
    try:
        agent = PolicyAgent(index, answerer, k=args.k, min_score=args.min_score)
        answer = agent.ask(args.question, parts=args.part or None)
    except RateLimited as err:
        raise SystemExit(f"policy-assistant: {err}") from None
    finally:
        index.client.close()
    _report_mode(answerer)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
    else:
        print(answer.render())
    return 0


def _report_mode(answerer) -> None:
    """Say on stderr how an OpenRouter model was asked, so CI logs show it."""
    mode = getattr(answerer, "last_mode", None)
    if mode == "structured":
        print(f"policy-assistant: {answerer.name} used structured outputs", file=sys.stderr)
    elif mode == "prompt":
        print(
            f"policy-assistant: {answerer.name} has no structured-output provider; "
            "it answered with JSON asked for in the prompt",
            file=sys.stderr,
        )


def _eval_build(args: argparse.Namespace) -> int:
    from .evaluate import build_eval, load_questions

    questions = load_questions(args.questions)
    try:
        summary = build_eval(args.corpus, args.out, questions)
    except ValueError as err:
        raise SystemExit(f"policy-assistant: {err}") from None
    print(json.dumps(summary, indent=2))
    return 0


def _eval_agent(args: argparse.Namespace) -> int:
    from .agent import PolicyAgent
    from .answerers import make_answerer
    from .evaluate import evaluate_agent, load_questions
    from .openrouter import RateLimited

    questions = load_questions(args.questions)[: args.limit]
    options = {"effort": args.effort} if args.answerer.startswith("claude") else {}
    try:
        answerer = make_answerer(args.answerer, **options)
    except ValueError as err:
        raise SystemExit(f"policy-assistant: {err}") from None
    index = _open_existing(args)
    try:
        agent = PolicyAgent(index, answerer, k=args.k, min_score=args.min_score)
        result = evaluate_agent(agent, questions)
        info, chunks = index.info(), index.count()
    except RateLimited as err:
        raise SystemExit(f"policy-assistant: {err}") from None
    finally:
        index.client.close()
    result = {
        "answerer": answerer.name,
        "k": args.k,
        "min_score": args.min_score,
        "encoder": info.get("encoder"),
        "chunks": chunks,
        **result,
    }
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", answerer.name)
    out = Path(args.out or f"results/agent__{slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out}", file=sys.stderr)
    return 0


def _eval_report(args: argparse.Namespace) -> int:
    from .evaluate import markdown_summary

    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.results]
    print(markdown_summary(results), end="")
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

    index = sub.add_parser("index", help="embed the corpus into Qdrant")
    index_sub = index.add_subparsers(dest="action", required=True)
    ibuild = index_sub.add_parser("build", help="(re)build the Qdrant collection")
    ibuild.add_argument("--corpus", default="data/corpus", help="folder with corpus.jsonl")
    ibuild.add_argument(
        "--model", default=DEFAULT_MODEL, help="fastembed model, or hashing-N (no download)"
    )
    ibuild.add_argument("--batch-size", type=int, default=128)
    _qdrant_args(ibuild)
    ibuild.set_defaults(func=_index_build)

    search = sub.add_parser("search", help="find the chunks closest to a question")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5, help="how many chunks to show")
    search.add_argument("--part", action="append", help="only this CFR part, e.g. 1005")
    search.add_argument("--model", help="encoder (default: the one that built the index)")
    _qdrant_args(search)
    search.set_defaults(func=_search)

    ask = sub.add_parser(
        "ask", help="answer a question with citations, or refuse if the text doesn't answer it"
    )
    ask.add_argument("question")
    ask.add_argument(
        "--answerer",
        default="claude",
        help="claude (default, needs ANTHROPIC_API_KEY), claude:<model id>, openrouter or "
        "openrouter:<model id> (needs OPENROUTER_API_KEY), or extractive",
    )
    ask.add_argument("--effort", default="medium", help="Claude's effort level")
    ask.add_argument("-k", type=int, default=5, help="how many chunks to retrieve")
    ask.add_argument("--min-score", type=float, default=0.0, help="drop chunks scoring lower")
    ask.add_argument("--part", action="append", help="only this CFR part, e.g. 1005")
    ask.add_argument("--json", action="store_true", help="print the answer as JSON")
    _qdrant_args(ask)
    ask.set_defaults(func=_ask)

    ev = sub.add_parser("eval", help="build the evaluation set and score the agent")
    ev_sub = ev.add_subparsers(dest="action", required=True)
    ebuild = ev_sub.add_parser("build", help="write the questions as a BEIR folder for RAGBench")
    ebuild.add_argument("--corpus", default="data/corpus", help="folder with corpus.jsonl")
    ebuild.add_argument("--out", default="data/regulations-qa", help="output folder")
    ebuild.add_argument("--questions", help="a questions.jsonl (default: the built-in set)")
    ebuild.set_defaults(func=_eval_build)

    eagent = ev_sub.add_parser("agent", help="ask the agent every question and score it")
    eagent.add_argument("--answerer", default="extractive", help="as for ask (default extractive)")
    eagent.add_argument("--effort", default="medium", help="Claude's effort level")
    eagent.add_argument("-k", type=int, default=5, help="how many chunks to retrieve")
    eagent.add_argument("--min-score", type=float, default=0.0, help="drop chunks scoring lower")
    eagent.add_argument("--questions", help="a questions.jsonl (default: the built-in set)")
    eagent.add_argument("--limit", type=int, help="only the first N questions")
    eagent.add_argument("--out", help="result JSON (default results/agent__<answerer>.json)")
    _qdrant_args(eagent)
    eagent.set_defaults(func=_eval_agent)

    ereport = ev_sub.add_parser("report", help="Markdown table of agent result files")
    ereport.add_argument("results", nargs="+")
    ereport.set_defaults(func=_eval_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
