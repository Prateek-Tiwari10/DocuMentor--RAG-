# DocuMentor

A RAG assistant that answers developer questions about **LangGraph** with cited, grounded answers. It is written in plain Python, without LangChain or LlamaIndex.

## Phase 1: ingestion and dense retrieval

Phase 1 builds the offline half of the system:

1. Download the LangGraph docs (a sparse git clone of `langchain-ai/docs`).
2. Clean each MDX page into two language views, Python and JS.
3. Chunk the views with one of two strategies: `recursive` (the baseline) or `parent_child` (the candidate).
4. Embed the chunks with `BAAI/bge-m3` and store them in ChromaDB, tagged with `language`.
5. Search the store, filtered by language.
6. Compare the two strategies on a 15-question mini golden set.

### Setup

```bash
source rag_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run

```bash
python scripts/download_docs.py
python scripts/ingest.py --strategy both --reset
python scripts/search.py "How do I add memory to a StateGraph?" --strategy both --k 3
python scripts/search.py "How do I add memory to a StateGraph?" --strategy parent_child --k 3 --language js
python scripts/eval_retrieval.py --strategy both --k 5
```

### Results

Mini golden set (15 Python questions), `BAAI/bge-m3`, k=5, run on 2026-10-11 after q01 gained a second accepted answer:

| Strategy | R@1 | R@3 | R@5 | MRR | Avg tokens returned |
|---|---|---|---|---|---|
| recursive | 0.67 | 0.80 | 0.80 | 0.722 | 2,011 |
| parent_child | 0.60 | 0.80 | 0.87 | 0.717 | 3,301 |

A result is a hit when it comes from an expected page and that entry's section appears in its heading path or as a heading inside the chunk. A question can list several accepted answers.

`--deep 50` shows where the misses rank:

| Question | recursive | parent_child |
|---|---|---|
| q05 (token streaming) | 6 | 31 |
| q07 (subgraph as a node) | 10 | 14 |
| q14 (InMemorySaver restarts) | 15 | hit at 1 |

### Layout

- `config/settings.yaml` holds every tunable number.
- `src/documentor/ingest/` has the loader, the MDX cleaner and the chunker.
- `src/documentor/embed/` has the sentence-transformers wrapper.
- `src/documentor/store/` has the Chroma collection wrapper and the JSON parent store.
- `src/documentor/retrieval/dense.py` searches both strategies.
- `eval/mini_golden.jsonl` is the mini golden set, and `eval/results/` holds the eval runs.

### Language views

Most LangGraph doc pages show each example twice, in a `:::python` block and a `:::js` block. The cleaner builds both views of every page. Each chunk carries `language: python | js`, and every search filters on one language, defaulting to Python.

### Notes

- `--fast` switches to `all-MiniLM-L6-v2`. That model truncates at **256 tokens**, so use it only to debug the pipeline, never to compare strategies.
- Tokens are counted with the `BAAI/bge-m3` tokenizer, never with character counts.
- Collections are named `<strategy>__<model_slug>`, for example `parent_child__BAAI-bge-m3`. Parents are stored in `data/parents/<collection>.json`.

## Phase 2 — Grounded generation

Phase 2 turns retrieval into an assistant. A question goes in, the top 5 `recursive` chunks are packed into a numbered context of at most 3,000 tokens, and `openai/gpt-oss-120b` on Groq answers from that context alone. Code then validates every citation against the context and lists only the cited sources. If the docs don't cover the question, the model replies `NOT_IN_DOCS` and the user gets a friendly refusal.

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then set GROQ_API_KEY
```

### Run

```bash
python scripts/ask.py "How do I add memory to a StateGraph?"
python scripts/ask.py "How do I add memory to a StateGraph?" --show-context
python scripts/ask.py "How do I add memory to a StateGraph?" --language js
```

To evaluate:

```bash
python scripts/eval_answers.py                 # RAG answers + automatic metrics + score calibration
python scripts/eval_answers.py --closed-book   # same questions, no retrieved docs
python scripts/eval_answers.py --summarize eval/results/phase2_<stamp>_review.md
```

### Results

All results below are from 2026-11-03, on 20 verified golden questions and 5 out-of-scope questions. `x01`–`x05` are the only questions tagged `post_cutoff: true`, each dated from the LangGraph changelog.

#### Retrieval

`BAAI/bge-m3`, k=5:

| Strategy | R@1 | R@3 | R@5 | MRR | Avg tokens returned |
|---|---|---|---|---|---|
| recursive | 0.60 | 0.75 | 0.75 | 0.667 | 1,991 |
| parent_child | 0.50 | 0.70 | 0.90 | 0.639 | 3,461 |

`--deep 50` shows where the misses rank:

| Question | recursive | parent_child |
|---|---|---|
| q05 (token streaming) | 6 | 31 |
| q07 (subgraph as a node) | 10 | 14 |
| q14 (InMemorySaver restarts) | 15 | hit at 1 |
| x03 (graceful shutdown) | 12 | hit at 4 |
| x04 (DeltaChannel) | >50 | hit at 4 |

#### Automatic metrics

Answers use the `recursive` strategy:

| Metric | Result |
|---|---|
| Answered (in-scope) | 90% (18 of 20) |
| Expected source cited | 70% (14 of 20) |
| Retrieved but not cited | q03 |
| Uncited answers | 0 |
| Invalid citations | 0 |
| Refusal accuracy (out-of-scope) | 100% (5 of 5) |
| Refused in-scope questions | q07, x04 |
| Mean tokens in / out | 1,915 / 218 |

Calibration: in-scope and out-of-scope top-1 scores overlap (lowest in-scope 0.574, highest out-of-scope 0.651), so `generation.min_score` stays off.

#### Latency

All 25 questions, with no cache hits:

| Measure | With waiting | Without waiting |
|---|---|---|
| p50 | 1,601 ms | 1,212 ms |
| p95 | 58,929 ms | 2,372 ms |

Waiting means rate-limiter pauses and retry sleeps, recorded as `wait_ms`. It averaged 18,399 ms per question, against 1,247 ms of generation (`generate_ms`) and 546 ms of retrieval. The first question also loads the embedding model, which took about 12 s and accounts for most of the retrieval average.

#### Answer grades

Grades from the review sheet: Y = correct and runnable, P = partly right or missing a key step, N = wrong or uses an API that doesn't exist.

All grades are by Prateek, who graded the closed-book answers and the q07 and x04 RAG answers directly. For the other 18 RAG answers, Claude drafted a grade by comparing each answer with its documentation section, without running the code, and Prateek reviewed every draft on 2026-11-05, keeping 12 and changing 6.

| Group | RAG Y | RAG P | RAG N | Closed-book Y | Closed-book P | Closed-book N | Closed-book outdated or invented API |
|---|---|---|---|---|---|---|---|
| All (20) | 9 | 8 | 3 | 0 | 1 | 19 | 19 |
| `post_cutoff: true` (5) | 1 | 3 | 1 | 0 | 0 | 5 | 5 |
| Untagged (15) | 8 | 5 | 2 | 0 | 1 | 14 | 14 |

### Findings

- **q07 and x04 are retrieval misses, refused because retrieval missed the section; graded N.** The expected section isn't in either context, so refusing was the right response to what the model saw. The metric still counts them as unanswered in-scope questions. The fix belongs in retrieval, not generation.
- **GPT-OSS writes its own citation format.** It often cites as `【1】` or `【1†L1-L7】` instead of `[1]`, and the first parser discarded those, so correct citations counted as missing. The parser now accepts both forms and rewrites them to `[n]`. The prompt now asks for `[n]` in the prose and never inside code, but 6 of 25 responses still used `【n】`, so the parser fix is needed as well.
- **`parent_child` finds more, `recursive` ranks better.** `parent_child` reaches R@5 0.90 against 0.75 for `recursive`, including q14, x03 and x04, but it returns about 74% more tokens and has a lower MRR.

### Free-tier limits

Groq's free plan allows 8,000 tokens per minute for this model, and one question uses about 2,100 tokens. Before each call, a token-per-minute limiter (`llm.tpm_limit`) reserves the prompt plus `max_completion_tokens`, so an eval run is paced instead of failing with 429s.

LLM responses are cached on disk in `data/cache/llm/`, keyed on the model, messages and sampling settings, so reruns cost nothing. `--no-cache` skips reading the cache but still writes to it. This cache is for development and eval reruns. A production cache comes in Phase 7.

### Layout

- `src/documentor/generation/llm.py` has the `LLM` interface, the Groq client with retries and rate limiting, and a `FakeLLM` for tests.
- `src/documentor/generation/prompt.py` builds the numbered context and the messages.
- `src/documentor/generation/citations.py` parses and validates citations and detects refusals.
- `src/documentor/pipeline.py` has `answer()`, which logs every query to `data/logs/queries.jsonl`.
- `src/documentor/evaluation.py` has the hit rule shared by both eval scripts.
- `eval/oos.jsonl` has five questions the corpus can't answer.

## Phase 3 — Evaluation

Phase 3 adds a repeatable evaluation harness. One command scores retrieval, answers and refusals on a 30-question golden set, saves a baseline, and compares later runs against it. The judge for the LLM-scored metrics is `openai/gpt-oss-120b` on Groq, called through RAGAS.

### Metrics

| Metric | What it measures | Computed by |
|---|---|---|
| Faithfulness | Share of an answer's claims supported by the retrieved context | RAGAS 0.4.3 |
| Answer correctness | Agreement with the reference answer | RAGAS 0.4.3 |
| Answer relevancy | Whether the answer addresses the question | RAGAS 0.4.3 |
| Context precision | Whether useful blocks rank above useless ones (reference-based) | RAGAS 0.4.3 |
| Citation validity | Every `[n]` points at a block that exists | Own code (`evaluation/metrics.py`) |
| Citation coverage | The answer cites the expected section | Own code |
| Refusal correctness | Out-of-scope questions refused, in-scope ones answered | Own code |
| Recall@k and MRR | The expected section is retrieved, and how high | Own code |

RAGAS scores only answered in-scope questions. A refusal is scored by refusal correctness instead. RAGAS and its dependencies are in `requirements-eval.txt`, separate from the app's `requirements.txt`.

### Golden set

`eval/mini_golden.jsonl` now has 30 questions, each with a short `reference` answer that `build_golden.py` checks against its section (`auto_checked: true`). The 10 new questions, q16–q25, cover error codes, deployment config, testing, tracing and two `post_cutoff: true` features (q19 and q20) dated from the LangGraph changelog. They are marked `verified: false` because Claude wrote them and no person has checked them yet. Every new question's retrieval rank was recorded before it was added, and no question was chosen or reworded because of its rank.

### Judge calibration

`calibrate_judge.py` scores the Phase 2 graded answers with each candidate judge, searches for the answer-correctness cut-offs that best match the human Y/P/N grades, and trusts the judge only at ≥ 80% agreement and Cohen's kappa ≥ 0.60.

| Judge | Cut-offs (Y / P) | Agreement | Kappa | Answers |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | 0.80 / 0.70 | 71% | 0.50 | 38 |
| `openai/gpt-oss-20b` | 0.85 / 0.75 | 66% | 0.42 | 35 (3 refused to score) |

Neither judge reaches the bar, so `calibrated: false`. `gpt-oss-120b` is kept as the judge because it agrees more often and never refused to score. Agreement splits sharply by answer type: 95% on the 20 closed-book answers, which are almost all N, but only 44% (kappa 0.10) on the 18 answered RAG answers. Answer correctness is therefore reported but not used for any decision.

The labels are the Phase 2 grades: Prateek graded the closed-book answers himself, and for the RAG answers he reviewed Claude's drafts. At 38 answers, one disagreement moves agreement by about 2.6 points.

### Results

Baseline run on 2026-11-07, `recursive`, dense retrieval (the default at the time), 30 golden and 5 out-of-scope questions. To save Groq budget, this run scored faithfulness only. The other three RAGAS metrics, closed-book RAGAS and `parent_child` answers were skipped.

| Config | Faithfulness | Citation validity | Citation coverage | Refusal correctness | R@5 / MRR | Tokens per answer | p50 generate ms |
|---|---|---|---|---|---|---|---|
| recursive | 0.938 | 1.000 | 0.821 (23 of 28) | 5/5 out-of-scope refused, 2 false refusals (q07, x04) | 0.83 / 0.739 | 2,145 | 1,163 |

- **Faithfulness:** 19 of 28 answers score 1.0. The lowest are q12 and q25 at 0.57, and q11 at 0.80.
- **Citation coverage misses:** q03, q05, q14, x03 and q21 answered without citing their expected section.
- **Reproducible:** rerunning from the cache with `--compare` against the baseline showed no changes.
- **Cost:** 145,782 judge tokens and 74,473 answer tokens, spread over two days of Groq's free daily budget.

### Strategy decision

The default stays `recursive`. The answer-level comparison with `parent_child` was deferred to Phase 4, after the reranker. The evidence so far is retrieval only: `parent_child` finds more (R@5 0.90 against 0.75 on the 20-question set) but returns about 75% more tokens.

### RAG vs closed-book

From the Phase 2 human grades, not from the judge:

| Group | RAG Y / P / N | Closed-book Y / P / N |
|---|---|---|
| All questions (20) | 9 / 8 / 3 | 0 / 1 / 19 |
| `post_cutoff: true` (5) | 1 / 3 / 1 | 0 / 0 / 5 |

19 of 20 closed-book answers used an outdated or invented API.

### Run

```bash
pip install -r requirements-eval.txt
python scripts/build_golden.py
python scripts/calibrate_judge.py
python scripts/eval_all.py --strategies recursive --metrics faithfulness --save-baseline
python scripts/eval_all.py --strategies recursive --compare eval/baselines/phase3.json
python scripts/eval_all.py --no-ragas              # own metrics only, no judge calls
```

`--compare` prints each metric old → new, the questions that got better or worse, and labels a one-question change "within noise". Judge scores are cached in `data/cache/ragas/`, so a rerun only pays for new work. When Groq's daily token budget runs out, the run stops with exit code 3 and prints "Daily Groq budget used up. Rerun the same command after …". Rerunning resumes from the cache. Groq's daily limit refills gradually, not at a fixed reset time, so a rerun right after the printed time fits only a few more calls.

### Layout

- `src/documentor/evaluation/` is now a package: `golden.py` loads and checks the golden set, `metrics.py` has the own metrics, `ragas_eval.py` has the RAGAS scorer, cache and budget stop, `agreement.py` has agreement, kappa and the cut-off search, and `report.py` builds the report and the comparison.
- `scripts/eval_all.py` is the one evaluation command. `eval/baselines/phase3.json` is the saved baseline.
