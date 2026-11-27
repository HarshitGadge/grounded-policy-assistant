# Grounded Policy Assistant

An agentic RAG assistant for U.S. regulations. It answers a question only from the
regulation text it retrieves, cites the exact paragraph each claim rests on
(for example `12 CFR 1005.6(b)(1)`), and says it can't answer when the text doesn't
support one. Retrieval quality is measured with [RAGBench](https://github.com/HarshitGadge/ragbench).

> Status: early. This first release builds the citable regulation corpus. Qdrant
> retrieval, the LangGraph agent and the RAGBench evaluation come in the next PRs.
> This README will only show numbers produced by runs in this repository.

## The corpus

The text comes from the [electronic Code of Federal Regulations](https://www.ecfr.gov)
(eCFR), a U.S. government work in the public domain. Every build is pinned to an eCFR
date, so the same date and parts always give the same text.

By default the corpus holds four consumer-finance regulations from the CFPB, with
their appendices and official interpretations:

| Part | Regulation | Covers |
| --- | --- | --- |
| 12 CFR 1002 | Regulation B | Equal credit opportunity |
| 12 CFR 1005 | Regulation E | Electronic fund transfers, card and transfer errors |
| 12 CFR 1016 | Regulation P | Privacy of consumer financial information |
| 12 CFR 1030 | Regulation DD | Truth in savings |

### How text becomes citable chunks

1. **Paragraph labels.** eCFR publishes each paragraph with its own marker, such as
   `(b)(1)` or `(ii)`. A small state machine follows the CFR's outline levels
   `(a)` → `(1)` → `(i)` → `(A)` and gives every paragraph its full label. Markers that
   could be a letter or a roman numeral, like `(i)` after `(h)(1)`, are settled by the
   marker that follows.
2. **Chunks follow the regulation's structure.** A chunk never mixes two top-level
   paragraphs such as `(a)` and `(b)`, and it is split further when it passes
   `--max-chars` (1,500 by default).
3. **Each chunk is cited by the deepest paragraph that holds all of its text**, so a
   chunk with `(b)(1)(i)` and `(b)(1)(ii)` is cited as `12 CFR 1005.6(b)(1)`.
4. **Appendices and interpretations.** Appendix chunks follow the appendix's own
   headings, and official interpretations are cited the way the CFPB cites them, as
   `comment 6(b)(1)-2`.
5. **Notes are left out.** Source citations, editorial notes and footnotes are not
   regulation text and are skipped.

Each chunk's text starts with its citation and section heading, so a retrieved chunk
says where it comes from.

## Quick start

```bash
pip install -e '.[dev]'
policy-assistant corpus build --date latest          # Regulations B, E, P and DD
policy-assistant corpus build --date 2026-11-01 --part 12:1026 --out data/reg-z
policy-assistant corpus stats data/corpus --show 3
```

The build writes three things to `--out` (default `data/corpus`):

- `corpus.jsonl`: one chunk per line in the [BEIR](https://github.com/beir-cellar/beir)
  corpus layout (`_id`, `title`, `text`), plus `citation`, `section`, `paragraphs` and
  `date`. RAGBench and other BEIR tools read it as is.
- `manifest.json`: the eCFR date, each part's source URL and SHA-256, and the counts.
- `raw/`: the downloaded XML, which a rebuild reuses instead of downloading again.

## Tests

```bash
pytest                  # offline, against a fixture shaped like eCFR's XML
```

CI runs lint, the tests on Python 3.10 to 3.13, a package install check, and a live
job that builds the default corpus from eCFR and checks it.

## License

MIT
