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
pytest -q
```

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
