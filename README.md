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

## Phase 4 — Hybrid retrieval

Phase 4 adds a keyword retriever (BM25) next to the dense one, merges the two lists with Reciprocal Rank Fusion (RRF), and can rerank the merged list with a cross-encoder:

`query → dense top 50 + BM25 top 50 → RRF → top 20 → cross-encoder → top 5 (→ parents) → context`

Ten setups were compared, and the default is now **`recursive` · hybrid, no reranker**. The reranker step and the parent step are skipped in this setup.

### Run

```bash
python scripts/build_bm25.py --strategy both     # rebuild after every ingest
python scripts/search.py "How do I use add_conditional_edges?" --mode hybrid --explain
python scripts/search.py "How do I use add_conditional_edges?" --mode hybrid --rerank --reranker fast --explain
python scripts/eval_retrieval.py --matrix --k 5 --deep 50 --device cpu
python scripts/eval_retrieval.py --apply-decision eval/results/phase4_<stamp>.json
```

`--mode` is `dense`, `bm25` or `hybrid`; `--rerank` adds a cross-encoder (`full` is `BAAI/bge-reranker-v2-m3`, `fast` is `cross-encoder/ms-marco-MiniLM-L-6-v2`). `--explain` prints each result's dense rank, BM25 rank, fused score and rerank score. The BM25 index lives in `data/bm25/`, one JSON file per collection and language. It indexes the same heading-prefixed text that is embedded. Words are lowercased and stopwords dropped, and an identifier is also split into its parts, so `add_conditional_edges` and `StateGraph` match `conditional` and `state`.

| Index | Chunks | Vocabulary |
|---|---|---|
| `recursive` · Python | 599 | 5,494 |
| `recursive` · JS | 562 | 5,146 |
| `parent_child` · Python | 1,357 | 5,494 |
| `parent_child` · JS | 1,242 | 5,144 |

### Results

Run on 2026-11-05 on a Kaggle CPU machine (4 cores), 30 golden questions, k=5, corpus 2caf896. R@ctx counts a hit only if the expected block survives the 3,000-token context budget, because a block that is cut never reaches the model.

| # | Config | R@5 | R@ctx | MRR | Tokens in context | p95 ms |
|---|---|---|---|---|---|---|
| 1 | recursive · dense (old default) | 0.83 | 0.83 | 0.739 | 1,571 | 450 |
| 2 | recursive · bm25 | 0.80 | 0.80 | 0.671 | 1,586 | 16 |
| **3** | **recursive · hybrid (new default)** | **0.93** | **0.93** | **0.780** | **1,591** | **478** |
| 4 | recursive · hybrid · fast reranker | 0.83 | 0.83 | 0.778 | 1,566 | 3,687 |
| 5 | recursive · hybrid · full reranker | 0.90 | 0.90 | 0.817 | 1,628 | 70,193 |
| 6 | parent_child · dense | 0.93 | 0.87 | 0.672 | 2,343 | 367 |
| 7 | parent_child · bm25 | 0.87 | 0.87 | 0.694 | 2,446 | 17 |
| 8 | parent_child · hybrid | 0.93 | 0.93 | 0.742 | 2,324 | 374 |
| 9 | parent_child · hybrid · fast reranker | 0.97 | 0.97 | 0.803 | 2,340 | 2,704 |
| 10 | parent_child · hybrid · full reranker | 0.97 | 0.97 | 0.829 | 2,348 | 51,466 |

Identifier questions name an API, error code or file, such as `get_stream_writer` or `langgraph.json` (12 of 30):

| Config | Identifier R@ctx (12) | Other R@ctx (18) |
|---|---|---|
| recursive · dense | 0.92 | 0.78 |
| recursive · bm25 | 0.83 | 0.78 |
| recursive · hybrid | 1.00 | 0.89 |
| parent_child · hybrid · fast reranker | 1.00 | 0.94 |

### Decision

The rule keeps the simpler setup unless a more expensive one gains at least 2 questions on R@ctx or 0.05 on MRR, without losing 2 questions or 0.05 MRR, and a reranking setup must stay under 2,000 ms at p95 on CPU. On 30 questions one question is 3.3 points, which is noise.

**recursive · hybrid: R@ctx 0.83 → 0.93 (+3 questions), MRR 0.739 → 0.780, p95 478 ms.** Rejected:

- recursive · bm25: lower (R@ctx −1 question, MRR −0.068).
- Every reranker setup: CPU p95 of 2,704–70,193 ms is over the 2,000 ms cap. Even ignoring the cap, the best of them, parent_child · hybrid · full, gains only 1 question on R@ctx over recursive · hybrid, so none would have passed the gain rule.
- parent_child · dense and parent_child · bm25: lower (R@ctx −2 questions).
- parent_child · hybrid: no clear gain (R@ctx +0, MRR −0.038), with about 46% more tokens in context.

**A question BM25 rescued:** x03 (graceful shutdown) ranked 12th for dense search and 1st for BM25, so the merged list puts it at 4. q07 (dense 10, BM25 3) and q14 (dense 15, BM25 9) were rescued the same way. No question fell out of the top 5, but q18 dropped from rank 1 to 4 and x02 from 2 to 5.

**Still missed:** q05 (token streaming) ranks 11th and x04 (DeltaChannel) ranks 29th in the merged list.

**Score threshold:** the hybrid top-1 score is an RRF score, and golden and out-of-scope questions overlap (lowest golden 0.028, highest out-of-scope 0.032), so `generation.min_score` stays off.

### End-to-end check

With the new default, on 2026-11-07:

- "Why is all my conversation history gone after restarting the server when I use InMemorySaver?" (q14, which dense search missed) is answered, and the answer cites the expected *Persistence > Troubleshooting common issues* block. That block was dense rank 15 and BM25 rank 9, and the merged list puts it 5th.
- "How do I add memory to a StateGraph?" is answered with checkpointer and store examples, citing two blocks from the memory page.
- "How do I create an evaluation dataset in LangSmith?" is refused, as it should be.

### Findings

- **BM25 alone is worse than dense, but the two miss different questions.** That is why fusion gains 3 questions while each retriever on its own is at 0.80–0.83.
- **The rerankers are too slow on CPU.** The full reranker takes about 55 s per query on 4 cores, so it needs a GPU to be practical.
- **The fast reranker hurts `recursive`.** It drops R@ctx from 0.93 to 0.83 (q07, q14 and x02 fall out). It was trained on web search passages (MS MARCO), which may explain why it ranks code docs worse than the fused order.

### Layout

- `src/documentor/retrieval/bm25.py` builds, saves and searches the BM25 index.
- `src/documentor/retrieval/fusion.py` has Reciprocal Rank Fusion.
- `src/documentor/retrieval/rerank.py` wraps the cross-encoders.
- `src/documentor/retrieval/hybrid.py` wires dense, BM25, fusion, reranking and parent grouping together.
- `scripts/build_bm25.py` builds the index, and `scripts/eval_retrieval.py --matrix` runs the comparison.

## Phase 5 — Conversations, language routing and index updates

Phase 5 makes DocuMentor conversational, routes each question to the Python or JS docs automatically, moves the vectors to Qdrant behind a store interface, and updates the index without re-embedding unchanged pages:

`message → language → standalone question → hybrid retrieval → answer`

History is used only to rewrite a follow-up. The answer model still sees one standalone question, with the unchanged Phase 2 prompt.

### Run

```bash
python scripts/chat.py                              # interactive; /lang python|js|auto, /reset, /history, /exit
python scripts/chat.py --script eval/demo_chat.txt  # one message per line, for demo rehearsal
python scripts/build_language_vocab.py
python scripts/eval_conversations.py
python scripts/migrate_to_qdrant.py                 # copy Chroma's vectors into Qdrant; --check compares both
python scripts/update_index.py --init               # once, for an existing index
python scripts/update_index.py [--no-pull] [--dry-run]
```

`ask.py` stays single-question.

### Follow-up questions

`eval/conversations.jsonl` has 12 two-turn conversations: 9 follow-ups that can't be understood without turn 1, and 3 topic switches. Turn 1 is a golden question. The set was drafted by an LLM from the cleaned Python pages, covering 10 pages, and every expected section was checked automatically against the docs. No person has verified it. Turn-2 retrieval with the default config, on 2026-11-07:

| Mode | R@ctx | MRR | Follow-up R@ctx (9) | Topic-switch R@ctx (3) | Rewrite p95 |
|---|---|---|---|---|---|
| `none` (the raw message) | 0.58 (7) | 0.444 | 0.56 (5) | 0.67 (2) | — |
| `concat` (previous question + message) | 0.67 (8) | 0.274 | 0.78 (7) | 0.33 (1) | — |
| `llm` (`gpt-oss-20b` rewrite) | 0.58 (7) | 0.486 | 0.56 (5) | 0.67 (2) | 666 ms |

**Decision: `none`.** The Phase 4 rule needs a gain of 2 conversations or 0.05 MRR without a loss. `concat` gains 1 conversation but loses 0.171 MRR, because the old question pushes the right block down. It also loses c10, a topic switch that the raw message finds at rank 1. `llm` gains 0.042 MRR and no conversations.

- **The LLM rewriter is too cautious.** It returned 4 of 9 follow-ups word for word ("Which checkpointer should I use for that in production?") and only dropped "And" from a fifth, which the prompt's "return it unchanged if already standalone" rule allows.
- **A good rewrite:** "How do I use it from inside a graph node?" became "How do I use an InMemoryStore with embeddings from inside a LangGraph node?", which moved a miss to rank 2.
- **A bad one:** adding "in LangGraph" to c09 pushed its section out of the top 5.
- **Raw follow-ups often work anyway,** because BM25 matches their content words ("restart", "thread", "delete").

Each rewrite costs about 450 tokens on `gpt-oss-20b`, which has its own rate limit, separate from the answer model's.

### Language routing

`detect_language()` is rule-based and returns a reason with every decision. It checks, in order:

1. A language passed by the caller, or set with `/lang`.
2. Explicit words: TypeScript, JavaScript, npm, Node.js, `@langchain/langgraph`, Python, pip and others.
3. Code identifiers that appear in only one view of the docs.
4. The conversation's previous language.
5. Python, the default.

The identifier lists are built from the cleaned docs (`data/language_vocab.json`): **576 Python-only** identifiers (`add_edge`, `TypedDict`, `add_conditional_edges`, ...) and **510 JS-only** ones (`addNode`, `StateSchema`, `addConditionalEdges`, ...). Identifiers in both views, such as `StateGraph` or `thread_id`, are neutral.

**Golden-set check:** all 30 golden questions route to Python. The spec's word list included `node`, which would have sent 6 of them to JS ("pick the next node", "unit test one node"), because "node" is core LangGraph vocabulary. Only `node.js` and `nodejs` count.

### Qdrant

`VectorStore` has two backends: `ChromaStore` and `QdrantStore` (local mode at `data/qdrant`; set `store.qdrant_url` for a server). The migration copies ids, vectors, text and metadata from Chroma without re-embedding. All 8 collection-and-language counts matched, for example 599 recursive Python chunks in both. Qdrant point ids are the 16-hex chunk ids read as integers, and the original id stays in the payload.

Parity check on 30 golden and 5 out-of-scope questions:

| Check | Result |
|---|---|
| Dense top-5 overlap, recursive / parent_child | 1.000 / 0.994 |
| Largest score difference | 0.000001 |
| Default config (recursive · hybrid) on Chroma | R@ctx 0.933, MRR 0.780 |
| Default config on Qdrant | R@ctx 0.933, MRR 0.780 |

The one difference is oos04 on parent_child: Qdrant's exact local search found a chunk at rank 3 that Chroma's approximate index missed. The check passed, so **the default backend is now Qdrant**. Chroma stays supported.

### Incremental index updates

`data/index_manifest.json` stores a SHA-256 hash of every cleaned page view. Snippets are inlined during cleaning, so editing a snippet changes every page that imports it. `update_index.py` then works in five steps:

1. Pull the docs.
2. Clean and hash every view, and classify each one as added, changed, removed or unchanged.
3. Delete changed and removed pages from every collection and from the parent store. Deleting by page matters because a page that shrinks would otherwise leave its old high-index chunks behind.
4. Chunk and embed only the added and changed pages, through the same code as `ingest.py`.
5. Rebuild BM25 and the language lists, which takes seconds, and report golden questions whose expected section disappeared.

- **No-change run (`--no-pull`, right after `--init`):** 90 views unchanged, 0 embedding calls, 8.2 s.
- **Dry run against the latest docs (89b6d1b, 2026-11-07):** 14 views added, 50 changed, 8 removed and 32 unchanged. Upstream split `graph-api` into several pages and removed `use-functional-api`, `choosing-apis`, `install` and `thinking-in-langgraph`. The dry run reads a temporary copy, so the checkout stays at the indexed commit (2caf896), and the update hasn't been applied.

### Findings

- **The demo chat routes language correctly but shows the follow-up gap.** "How do I do the same thing in TypeScript?" routes to JS and answers in TypeScript. With no rewrite, it answered about long-term stores instead of the previous turn's checkpoint history. The LLM rewriter also returned that message unchanged.
- **Retrieval is unchanged on the new backend.** `eval_retrieval.py --k 5` gives R@5 0.933 and MRR 0.780 on recursive, as in Phase 4.

### Layout

- `src/documentor/language.py` builds the identifier lists and detects the language.
- `src/documentor/conversation.py` has `Turn`, `Conversation` and `make_standalone()`.
- `src/documentor/store/base.py` has the `VectorStore` protocol and `get_store()`, and `store/qdrant_store.py` has `QdrantStore`.
- `src/documentor/ingest/updater.py` plans and applies incremental updates.
