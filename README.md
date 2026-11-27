# Grounded Policy Assistant

An agentic RAG assistant for U.S. regulations. It answers a question only from the
regulation text it retrieves, cites the exact paragraph each claim rests on
(for example `12 CFR 1005.6(b)(1)`), and says it can't answer when the text doesn't
support one. Retrieval quality is measured with [RAGBench](https://github.com/HarshitGadge/ragbench).

> Status: early. It builds the citable regulation corpus, searches it in Qdrant,
> answers with citations or refuses, and is evaluated with RAGBench in CI.
> This README only shows numbers produced by runs in this repository.

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

## Search it in Qdrant

```bash
pip install -e '.[fastembed]'
policy-assistant index build --corpus data/corpus --qdrant data/qdrant
policy-assistant search "How long does a bank have to investigate a transfer error?" -k 5
policy-assistant search "Can a creditor ask about marital status?" --part 1002
```

- **Embeddings.** Chunks are embedded with
  [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) through
  [fastembed](https://github.com/qdrant/fastembed), which runs on the CPU with no API key.
  `--model` takes any other fastembed model. Each chunk is embedded with its citation line
  first, so a question that names a section can match it.
- **Qdrant.** One point per chunk, with cosine distance. The point id comes from the chunk
  id, so rebuilding the same corpus gives the same points. The payload holds the citation,
  section, part, eCFR date and text, so a result can be quoted and cited directly.
  `--qdrant` keeps the index in a local folder; `--url` uses a Qdrant server instead.
- **The index remembers its encoder.** The collection records the model that built it,
  `search` uses that model, and a search with a different one is refused, because vectors
  from two models can't be compared.
- **RAGBench.** `policy_assistant.retriever:QdrantRetriever` is a RAGBench retriever, so
  the same retrieval can be scored on any BEIR-format dataset:

  ```bash
  ragbench run --data path/to/beir-data \
      --retriever policy_assistant.retriever:QdrantRetriever
  ```

## Ask it

```bash
export ANTHROPIC_API_KEY=...       # each question is one or two billed API requests
policy-assistant ask "How much can a consumer lose if they report a lost debit card within two business days?"
policy-assistant ask "Can a creditor ask about marital status?" --part 1002 --json
policy-assistant ask --answerer extractive "What is a preauthorized transfer?"   # no model, no key

export OPENROUTER_API_KEY=...      # any model on OpenRouter; the default is a free one
policy-assistant ask --answerer openrouter "What is a preauthorized transfer?"
policy-assistant ask --answerer openrouter:<model id> "What is a preauthorized transfer?"
```

An answer is a list of short claims, each with numbered citations, then the sources:

```
<claim> [1]
<claim> [1][2]

Sources:
[1] 12 CFR <section and paragraph>
[2] 12 CFR part <part>, Supplement I to Part <part>, comment <number>
```

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) graph
(`policy_assistant.agent.PolicyAgent`):

```
retrieve -> grade --(nothing relevant)--------------------------> refuse
              \--> generate -> verify --(every claim cited)-----> answer
                       ^          |----(a claim lost its source)-> generate (once more)
                       |__________|----(the text doesn't answer)-> refuse
```

- **retrieve** takes the `-k` closest chunks; **grade** drops any scoring under
  `--min-score`, and refuses without calling a model if none are left.
- **generate** asks Claude (`claude-opus-5-5` by default; `--answerer claude:<model id>`
  picks another) for an answer in a fixed JSON schema: either claims that each cite the
  passages they rest on, or `answerable: false` with a reason. The schema only allows
  citations of the passages Claude was shown, and the prompt tells it to refuse rather
  than answer from its own knowledge.
- **verify** keeps a claim only if it cites a chunk that was retrieved. If it drops one,
  Claude gets one more try with a note saying which; if no claim survives, the agent
  refuses. So every citation in an answer points at text the agent actually read.
- A refusal says why: `no_passages`, `unsupported` (Claude said the text doesn't answer),
  `uncited`, or `model_refused` (Claude declined the request; the API's server-side
  fallback is on, so it first retries on Anthropic's recommended fallback model).

`--answerer openrouter` runs the same graph with a model on
[OpenRouter](https://openrouter.ai), through its OpenAI-compatible chat completions API,
with the same prompt and answer schema. The model is `nvidia/nemotron-3.5-super-vl:free`
unless `--answerer openrouter:<model id>` or `OPENROUTER_MODEL` names another. It asks
for structured outputs (a strict JSON schema, routed only to providers that support
it), and if the model has no such provider it falls back to asking for the JSON in the
prompt. OpenRouter limits free models to a few requests a day per account (50 a day,
according to its FAQ, unless the account has bought credits), retries are spaced out on
a rate limit, and free endpoints may keep prompts; the prompts here are public
regulation text and the question.

`--answerer extractive` swaps Claude for a baseline that quotes the single best-matching
sentence, or refuses when no sentence covers most of the question's words. It calls no
model, so the tests and CI can run the whole graph without a key; it is the floor that
Claude has to beat in the evaluation.

## Evaluation

The questions are hand-written and live in
[`src/policy_assistant/data/questions.jsonl`](src/policy_assistant/data/questions.jsonl):
49 in-scope questions across Regulations B, E, P and DD, each labeled with the one CFR
section that answers it, and 12 out-of-scope questions that nothing in the corpus
answers (some far off, like highway speed limits, some close, like check holds under
Regulation CC or credit card late fees under Regulation Z, which aren't in the corpus).

```bash
policy-assistant eval build --corpus data/corpus --out data/regulations-qa
ragbench run --data data/regulations-qa --retriever policy_assistant.retriever:QdrantRetriever \
    --param model=BAAI/bge-small-en-v1.5 --name qdrant-bge-small --out results/retrieval__qdrant.json
ragbench run --data data/regulations-qa --retriever ragbench.retrievers.bm25:BM25 \
    --name bm25 --out results/retrieval__bm25.json
ragbench report results/retrieval__*.json
policy-assistant eval agent --answerer extractive          # or claude, openrouter:<model id>
policy-assistant eval report results/agent__*.json
```

`eval build` turns the labels into relevance judgements over the built corpus: every
chunk of the labeled section, and every official interpretation of it in Supplement I,
counts as relevant. It fails if a labeled section has no chunk, so a wrong label can't
quietly lower the score. It writes a BEIR folder, so [RAGBench](https://github.com/HarshitGadge/ragbench)
scores retrieval with no conversion, with bootstrap confidence intervals.

Relevance is judged per section, and a section can have more relevant chunks than fit
in the top 10, so recall@10 can't reach 1 for those questions. MRR@10 (how high the first
relevant chunk ranks) is the measure that matters for the agent, which reads the top 5.

`eval agent` asks the agent every question. An in-scope question counts as correct when
the agent answers and cites at least one chunk of the labeled section; an out-of-scope
question counts as correct when it refuses. CI runs it with the extractive answerer
only, since the model answerers are billed per question (Claude) or limited to a few
requests a day (OpenRouter's free models).

### Results

From [CI run 46038100221](https://github.com/HarshitGadge/grounded-policy-assistant/actions/runs/46038100221)
(commit e56986d), on the default corpus built from eCFR in that run. The result files are
the run's `results` artifact. 95% bootstrap intervals are in parentheses.

Retrieval, scored by RAGBench on the 49 in-scope questions (40 sections, 1,050 relevant
chunks in all, 3 to 187 per question):

| retriever | nDCG@10 | recall@10 | recall@100 | MRR@10 | p50 ms |
|---|---|---|---|---|---|
| BM25 (RAGBench's) | 0.4505 (0.375 to 0.530) | 0.2634 (0.211 to 0.320) | 0.5966 (0.527 to 0.667) | 0.6872 (0.577 to 0.790) | 0.75 |
| Qdrant, bge-small-en-v1.5 | 0.5894 (0.526 to 0.653) | 0.3595 (0.294 to 0.427) | 0.7056 (0.637 to 0.770) | 0.8714 (0.785 to 0.946) | 12.87 |

`ragbench compare` (paired bootstrap, 49 questions): dense retrieval beats BM25 on
nDCG@10 by +0.1389 (95% CI +0.0914 to +0.1900), on recall@10 by +0.0961 (+0.0483 to
+0.1469) and on MRR@10 by +0.1842 (+0.0960 to +0.2729), all significant.

The agent with the extractive answerer (no model), k = 5:

| answerer | in-scope correct | answered | correct when answered | out-of-scope refused |
|---|---|---|---|---|
| extractive | 67% | 80% | 85% | 83% |

For 48 of the 49 in-scope questions (98%), the five retrieved chunks include one from
the labeled section, so most of the gap to 100% is the answerer, which only matches
words. Model answerers can be scored with the same command; their numbers will be
added here once they come from a run in this repository.

## Tests

```bash
pytest                  # offline: a fixture shaped like eCFR's XML, in-memory Qdrant
pytest -m live          # against a corpus and index you built (see tests/live)
```

The offline tests use a hashing encoder (`--model hashing-512`), which needs no model
download and only matches shared words, and stand-ins for the Anthropic and OpenRouter
clients, so they need no key. CI runs lint, the tests on Python 3.10 to 3.13, a package install
check, and a live job that builds the default corpus from eCFR, indexes it with the
real model, checks that sample questions retrieve the section that answers them, and
runs the agent with the extractive answerer. If the repository has an
`ANTHROPIC_API_KEY` secret, the live job also asks Claude a few questions and checks
that it cites the right section and refuses out-of-scope questions. An
`OPENROUTER_API_KEY` secret does the same with the OpenRouter model (the
`OPENROUTER_MODEL` repository variable, or the free default), and those checks are
skipped if the free model's daily limit is reached. Without the secrets these checks
are skipped.

## License

MIT
