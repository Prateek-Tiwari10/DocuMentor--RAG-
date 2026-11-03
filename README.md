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

## Phase 2: grounded generation

Phase 2 turns retrieval into an assistant. A question goes in, and a grounded answer with validated `[n]` citations comes out, or a refusal when the docs don't cover it.

- **Model:** `openai/gpt-oss-120b` on Groq with `reasoning_effort: low`. The model name lives only in `config/settings.yaml`.
- **Context:** the top 5 `recursive` results, capped at 3,000 tokens, best-ranked block first.
- **Citations:** code parses `[n]` outside code (fenced and inline), drops numbers that aren't in the context, and lists only the cited sources.
- **Refusal:** the model replies `NOT_IN_DOCS` and the pipeline maps it to a friendly message. A retrieval-score pre-check (`generation.min_score`) exists but stays off until calibrated.

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then set GROQ_API_KEY
```

### Run

```bash
python scripts/ask.py "How do I add memory to a StateGraph?" --show-context
python scripts/ask.py "How do I add memory to a StateGraph?" --language js
python scripts/eval_answers.py                 # RAG answers + automatic metrics + score calibration
python scripts/eval_answers.py --closed-book   # same questions, no retrieved docs
python scripts/eval_answers.py --summarize eval/results/phase2_<stamp>_review.md
```

### Free-tier limits

Groq's free plan allows 8,000 tokens per minute for this model, and one question costs about 5,000 tokens. Every call goes through a token-per-minute limiter (`llm.tpm_limit`), so an eval run is paced instead of failing with 429s.

LLM responses are cached on disk in `data/cache/llm/`, keyed on the model, messages and sampling settings, so reruns cost nothing. `--no-cache` skips reading the cache but still writes to it. This cache is for development and eval reruns. A production cache comes in Phase 7.

### Layout

- `src/documentor/generation/llm.py` has the `LLM` interface, the Groq client with retries and rate limiting, and a `FakeLLM` for tests.
- `src/documentor/generation/prompt.py` builds the numbered context and the messages.
- `src/documentor/generation/citations.py` parses and validates citations and detects refusals.
- `src/documentor/pipeline.py` has `answer()`, which logs every query to `data/logs/queries.jsonl`.
- `src/documentor/evaluation.py` has the hit rule shared by both eval scripts.
- `eval/oos.jsonl` has five questions the corpus can't answer.
