# Phase 1 — Ingestion and dense retrieval (recursive vs parent-child)

## Context

DocuMentor is a production-style RAG assistant that answers developer questions about **LangGraph** with cited, grounded answers. It is being built phase by phase as a resume project, in plain Python (no LangChain / LlamaIndex).

Phase 0 is done: embedding and cosine-similarity experiments.

Phase 1 builds the offline half of the system:

1. Download the LangGraph docs.
2. Clean them into two **language views** per page: a Python view and a JS view.
3. Chunk them with **two strategies** behind one config flag.
4. Embed the chunks and store them in ChromaDB, each tagged with `language` metadata.
5. Search the store, filtered by language (default Python).
6. Measure which strategy retrieves better on a 15-question mini golden set.

The LangGraph docs show most examples twice, once in a `:::python` block and once in a `:::js` block. Deleting the JS version would leave TypeScript users with no answers. Mixing both versions would put TypeScript into Python answers. So both are kept, every chunk is tagged with its language, and every search filters on one language. Python users never see JS chunks, and JS users get JS code.

If an older `ingest.py` / `search.py` exists from an earlier FastAPI-docs experiment, replace it. That corpus is no longer used.

The developer is learning while building. Keep each module small and readable so every step can be explained in an interview.

## Locked decisions (do not change)

| Area | Decision |
| --- | --- |
| Corpus | LangGraph docs from `github.com/langchain-ai/docs`. Folders `src/oss/langgraph/` and `src/oss/concepts/` (`.mdx`), plus `src/snippets/code-samples/` to resolve imported code samples. |
| Language views | Build a Python view and a JS view of every page. Shared text goes in both views; `:::python` blocks and `*-py.mdx` snippets go only in the Python view; `:::js` blocks and `*-js.mdx` snippets go only in the JS view. Every chunk and parent carries `language: python \| js`. Search always filters on one language, defaulting to `python`. |
| Chunking A (baseline) | Recursive: ~500 tokens, ~75-token overlap, split on headings → paragraphs → lines → sentences → words. |
| Chunking B (candidate) | Parent-child. Parent = H2 section (with its H3+), capped at ~1,500 tokens. Child = ~200 tokens with ~30-token overlap. Embed children; return parents. |
| Code blocks | A fenced code block is never split. An oversized block becomes its own chunk. |
| Token counting | Use the `BAAI/bge-m3` tokenizer (`transformers.AutoTokenizer`), never character counts. |
| Embeddings | `BAAI/bge-m3` via `sentence-transformers`, `normalize_embeddings=True`. `all-MiniLM-L6-v2` only as a `--fast` debug option. |
| Vector DB | ChromaDB `PersistentClient`, cosine space, one collection per strategy + model. |
| Parent storage | Parents in a JSON file (`data/parents/<collection>.json`), keyed by `parent_id`. Children carry `parent_id` in metadata. |
| Orchestration | Plain Python. No LangChain, LlamaIndex or their text splitters. |

## Environment

Use the existing `rag_env` virtualenv. Add `requirements.txt`:

```
sentence-transformers>=3.0
transformers>=4.40
chromadb>=0.5
pyyaml
tqdm
pytest
```

Git 2.27+ is needed for sparse checkout. The commands below work in Git Bash or PowerShell on Windows, as well as on Linux and macOS.

## Target layout

```
documentor/
├── config/settings.yaml
├── data/                         # gitignored
│   ├── raw/langchain-docs/       # sparse clone
│   ├── raw/manifest.json
│   ├── clean/                    # cleaned .md per page
│   ├── chroma/
│   └── parents/
├── scripts/
│   ├── download_docs.py
│   ├── ingest.py
│   ├── search.py
│   └── eval_retrieval.py
├── src/documentor/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest/{__init__,loader,cleaner,chunker}.py
│   ├── embed/{__init__,embedder}.py
│   ├── store/{__init__,chroma_store,parent_store}.py
│   └── retrieval/{__init__,dense}.py
├── eval/
│   ├── mini_golden.jsonl
│   └── results/
├── tests/{test_cleaner,test_chunker,test_retrieval}.py
├── requirements.txt
├── .gitignore
└── README.md
```

Make `src/documentor` importable: either add a minimal `pyproject.toml` and run `pip install -e .`, or put a `sys.path` insert at the top of each script. Prefer the `pyproject.toml` route.

## Config — `config/settings.yaml`

```yaml
corpus:
  repo: https://github.com/langchain-ai/docs.git
  branch: main
  paths: [src/oss/langgraph, src/oss/concepts, src/snippets/code-samples]
  page_dirs: [src/oss/langgraph, src/oss/concepts]
  snippet_dir: src/snippets/code-samples
  raw_dir: data/raw/langchain-docs
  clean_dir: data/clean
  url_base: https://docs.langchain.com/oss   # + /python/... or /javascript/...
  languages: [python, js]
  default_language: python
  url_segment: {python: python, js: javascript}

embedding:
  model: BAAI/bge-m3
  fast_model: sentence-transformers/all-MiniLM-L6-v2
  tokenizer: BAAI/bge-m3
  batch_size: 16
  device: auto            # auto -> cuda if available else cpu

chunking:
  strategy: parent_child  # recursive | parent_child
  prepend_headings: true  # prefix "Page > H2 > H3" to the text that gets embedded
  recursive:
    size: 500
    overlap: 75
  parent_child:
    child_size: 200
    child_overlap: 30
    parent_max: 1500
    child_top_k: 20       # children fetched before grouping into parents

store:
  chroma_path: data/chroma
  parents_path: data/parents

retrieval:
  top_k: 5
```

`src/documentor/config.py` loads this file once and returns a dict, using `functools.lru_cache`. CLI flags may override `chunking.strategy` and the model.

## Step-by-step tasks

### 1. Download the docs — `scripts/download_docs.py`

Run the equivalent of these commands, skipping the clone if the folder already exists and running `git pull` instead:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/langchain-ai/docs.git data/raw/langchain-docs
cd data/raw/langchain-docs
git sparse-checkout set src/oss/langgraph src/oss/concepts src/snippets/code-samples
git rev-parse HEAD
```

Use `subprocess.run(..., check=True)`. Then write `data/raw/manifest.json`:

```json
{"repo": "...", "commit": "<full hash>", "downloaded_at": "<ISO time>",
 "pages": ["src/oss/langgraph/add-memory.mdx", "..."], "page_count": 0, "snippet_count": 0}
```

Pages are every `.mdx` / `.md` file under `page_dirs`, recursively (this includes `langgraph/errors/` and `langgraph/frontend/`).

**Acceptance:** about 50 pages and about 700 snippet files. The script prints the commit hash. A second run is fast and does not re-clone.

Manual fallback, if git fails: download the repo ZIP from GitHub and copy the three folders into `data/raw/langchain-docs/` at the same paths. Write the manifest with `commit: "zip-<date>"`.

### 2. Clean MDX into two language views — `src/documentor/ingest/cleaner.py`

Expose `clean_page(text: str, snippets: dict[str, str], language: str) -> tuple[str, dict]`. `language` is `"python"` or `"js"`. It returns the cleaned Markdown for that language's view and metadata (`title`, plus anything else from the frontmatter). The ingest step calls it once per language for every page.

The format was observed in the repo at commit `33ac8ab`. Handle all of the following.

1. **Frontmatter:** parse the leading `---` … `---` block. Keep `title`, then remove the block.
2. **Snippet imports:** lines like `import LanggraphInterruptsResumeV2Py from '/snippets/code-samples/langgraph-interrupts-resume-v2-py.mdx';`.
   - Build a map from component name to snippet file.
   - In the Python view, replace each usage such as `<LanggraphInterruptsResumeV2Py />` with that snippet's content if the file ends in `-py.mdx`, and with nothing if it ends in `-js.mdx`. The JS view does the opposite.
   - A snippet with neither suffix goes in both views.
   - Remove all `import` and `export` lines that sit outside code fences.
   - There are about 113 such imports across the pages.
3. **Language blocks:** a line that is only `:::python` or `:::js` opens a block, and a line that is only `:::` closes it. Markers can be indented, for example inside `<Tabs>`.
   - In the Python view, drop `:::js` blocks entirely, including their content. For `:::python` blocks, keep the content and remove the marker lines.
   - In the JS view, do the opposite: drop `:::python` blocks and keep the content of `:::js` blocks.
   - Text outside any language block is shared and goes in both views.
   - Track the block with a small state machine and match on stripped lines.
4. **JSX components:** outside code fences only.
   - Remove opening and closing tags such as `<Tabs>`, `<Tab title="Postgres">`, `<Tip>`, `<Note>`, `<Warning>`, `<Info>`, `<Accordion title="...">`, `<CodeGroup>`, `<Steps>`, `<Step title="...">`, `<Card ...>`, `<CardGroup>`, `<Icon .../>` and self-closing components. Keep the inner text.
   - When a tag has a `title="..."` attribute, emit the title as a bold line (`**Postgres**`) so the context survives.
   - Remove empty `<Tabs>` / `<CodeGroup>` shells left behind after the other language's blocks are removed.
5. **Code-fence annotations:** remove trailing `# [!code highlight]`, `// [!code highlight]` and similar `[!code ...]` comments inside fences.
6. **Never touch code-fence contents** apart from rule 5. Python code contains `<`, `>` and `:::`-like text.
7. **Whitespace:** dedent content that was only indented because it sat inside a JSX tag. Collapse three or more blank lines into one blank line.
8. **Links:** keep link text and target, and turn relative links like `/oss/langgraph/persistence` into full URLs on `https://docs.langchain.com`, using the view's language segment (`/oss/python/...` or `/oss/javascript/...`). These help later with citations.

`src/documentor/ingest/loader.py`:

- `load_snippets(cfg) -> dict[str, str]` maps file stem to content.
- `load_pages(cfg) -> list[dict]` returns one dict per page **per language**, with keys `{"path", "slug", "language", "url", "title", "text"}`.
- `slug` is the page path relative to `src/oss/` without the extension, for example `langgraph/add-memory`. `url` is `url_base + "/" + url_segment[language] + "/" + slug`.
- Save each cleaned view to `data/clean/<language>/<slug>.md` so the output can be read by eye.
- Skip views whose cleaned text is under 50 tokens, and log them.
- Pages with no language blocks produce two identical views. That's fine: search filters on one language, so the duplicate never appears twice in one result list.

**Acceptance:**

- `grep -rE ":::js|typescript|@langchain/langgraph" data/clean/python` finds no matches outside rare prose mentions.
- `grep -rE ":::python|^\`\`\`python" data/clean/js` finds no matches.
- No line in `data/clean` starts with `<Tab` or `import `.
- Spot-check `add-memory.md` in both views: the Python view has the `InMemorySaver` example and no TypeScript; the JS view has the `MemorySaver` TypeScript example and no Python. Both contain the same paragraphs of shared text.

### 3. Chunking — `src/documentor/ingest/chunker.py`

#### Shared helpers

**Tokenizer:**

- `count_tokens(text)` uses the `BAAI/bge-m3` tokenizer, loaded once with `add_special_tokens=False`.

**Parsing:**

- `parse_blocks(markdown) -> list[Block]` splits a page into ordered blocks.
- `Block` is a dataclass with fields `kind` (`heading` / `code` / `text`), `level` (for headings), `text`, and `heading_path` (a list such as `["Memory", "Add short-term memory", "Use in production"]`).
- A whole fenced code block, from its opening fence through the matching closing fence, is one `code` block.
- Paragraphs, lists and tables become `text` blocks, separated by blank lines.

**Splitting oversized text:**

- `split_text(text, max_tokens)` recursively splits a single oversized `text` block.
- Try separators in this order: `"\n"`, then sentence boundaries (`". "`, `"? "`, `"! "`), then `" "`.

**Packing with overlap:**

- `pack(blocks, size, overlap)` greedily packs blocks into chunks of at most `size` tokens.
- When a chunk closes, the next one starts with the tail of the previous chunk, about `overlap` tokens. Take whole sentences from the last text block, and never a partial code block. If the last block was code, carry no overlap.
- A `code` block larger than `size` becomes its own chunk. Log how many there are.

**Chunk record:**

Every chunk is a dict with these keys:

```python
{"id", "text", "embed_text", "page_slug", "language", "url", "title", "heading_path",
 "tokens", "strategy", "parent_id"}  # parent_id is None for recursive
```

- `embed_text` is `"{title} > {' > '.join(heading_path)}\n\n{text}"` when `prepend_headings` is true, and `text` otherwise. It is used for both strategies, so the comparison stays fair.
- `heading_path` is stored in metadata as a `" > "`-joined string, because Chroma metadata must be scalar.
- IDs are deterministic: `sha1(f"{strategy}|{language}|{page_slug}|{index}")[:16]`. Re-ingesting then overwrites chunks instead of duplicating them, and the two language views never collide.
- Chunking runs on one language view of one page at a time, so a chunk never mixes languages.

#### Strategy A — `chunk_recursive(page, cfg) -> list[dict]`

- Parse the blocks of one page and pack them with `size=500` and `overlap=75`.
- Prefer to close a chunk at a heading boundary. If adding the next H2 or H3 heading would push the chunk past about 60% of `size`, close the chunk first.
- Chunks never cross page boundaries.

#### Strategy B — `chunk_parent_child(page, cfg) -> tuple[list[dict], list[dict]]`

**Parents:**

- One parent per H2 section, including its nested H3 and deeper sections.
- Content before the first H2 becomes an intro parent whose heading path is just the page title.
- If a parent exceeds `parent_max` (1,500) tokens, split it at H3 boundaries.
- If an H3 piece is still too big, split it with `pack(size=1500, overlap=0)`.
- Parent record: `{"parent_id", "text", "page_slug", "language", "url", "title", "heading_path", "tokens"}`.
- `parent_id` = `sha1(f"parent|{language}|{page_slug}|{index}")[:16]`.

**Children:**

- Split each parent with `pack(size=200, overlap=30)`.
- Every child stores its parent's `parent_id`.
- Children never cross parent boundaries.

The function returns `(children, parents)`.

**Acceptance:**

- No chunk has an odd number of ```` ``` ```` fence lines.
- Every child's `parent_id` exists among the parents, and the child and parent have the same `language`.
- Except for logged oversized code chunks: recursive chunks are ≤ 500 tokens, children ≤ 200 tokens, parents ≤ 1,500 tokens.
- Consecutive recursive chunks on the same page share overlapping text, unless the boundary was a code block.

### 4. Embedding — `src/documentor/embed/embedder.py`

- `class Embedder` takes `__init__(model_name, device, batch_size)` and provides `embed_documents(texts) -> list[list[float]]` and `embed_query(text) -> list[float]`.
- Use `SentenceTransformer(model_name, device=...)` and `encode(..., normalize_embeddings=True, batch_size=..., show_progress_bar=True)`.
- bge-m3 needs no query instruction prefix. Keep the method separate anyway, so later models that need a prefix fit in.
- `device: auto` resolves to `cuda` if `torch.cuda.is_available()`, else `cpu`.
- Print the resolved device and model at start-up.
- Note in the README: `all-MiniLM-L6-v2` truncates at 256 tokens. Use it only for pipeline debugging, never for the strategy comparison.

### 5. Storage

**`src/documentor/store/chroma_store.py`:**

- `class ChromaStore(path, collection)` wraps `chromadb.PersistentClient(path)`.
- Use `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`.
- Methods:
  - `reset()` deletes and recreates the collection.
  - `add(chunks, embeddings)` upserts in batches of 1,000, with documents = `text` and metadata = every scalar field except `text` and `embed_text`. `language` must be in the metadata.
  - `query(embedding, k, language=None) -> list[dict]` returns `text`, metadata and `score = 1 - distance`. When `language` is given, pass `where={"language": language}` to Chroma so the filter runs inside the search, not after it. Filtering after the search would return fewer than `k` results.
  - `count()` returns the number of stored chunks.

**`src/documentor/store/parent_store.py`:**

- `class ParentStore(path)` provides `save(parents)` (writes JSON), `get(parent_id)` and `get_many(ids)`.

**Collection names:**

- `f"{strategy}__{model_slug}"`, where `model_slug` is the model name with `/` and `.` replaced by `-`.
- Examples: `recursive__BAAI-bge-m3` and `parent_child__BAAI-bge-m3`.
- Both languages live in the same collection and are separated by the `language` metadata filter. One collection per language is not needed.

### 6. Ingest CLI — `scripts/ingest.py`

```
python scripts/ingest.py --strategy recursive|parent_child|both [--languages python,js] [--fast] [--reset]
```

`--languages` defaults to `corpus.languages` from the config.

Flow: load config → load snippets and pages → clean each page once per language → chunk → embed `embed_text` → store (and save parents for `parent_child`).

Print a summary per strategy, broken down by language:

```
strategy=parent_child  language=python  pages=49  parents=312  children=1,104
avg_child_tokens=176  avg_parent_tokens=611  oversized_code=3
strategy=parent_child  language=js      pages=49  parents=305  children=1,087  ...
total_time=371s
```

Also write `data/ingest_stats_<strategy>.json` with the same numbers plus the corpus commit hash.

Optional, for slow local runs: the same script runs unchanged on a Kaggle GPU notebook. Zip `data/chroma` and `data/parents` and copy them back.

### 7. Retrieval — `src/documentor/retrieval/dense.py` and `scripts/search.py`

`search(query, strategy, k=5, language="python") -> list[dict]` behaves differently per strategy. `language` defaults to `corpus.default_language` and is always applied as a filter, for both strategies.

**Recursive:**

- Embed the query, then take the top `k` chunks of that language from the recursive collection.

**Parent-child:**

- Take the top `child_top_k` (20) children of that language.
- Walk them in rank order and collect unique `parent_id`s until there are `k` parents.
- Each result is the parent record plus `best_child_score`, `best_child_text` and `matched_children` (how many children hit that parent).

Every result includes `rank`, `score`, `language`, `url`, `heading_path` and `tokens`.

CLI:

```
python scripts/search.py "How do I add memory to a StateGraph?" --strategy both --k 3 [--language python|js]
```

- `--language` defaults to `python`.
- Print both strategies' results.
- Each result line shows: rank, score with 3 decimals, language, `page_slug`, `heading_path`, token count, and the first 200 characters of the text.

Automatic language detection from the question, for example switching to JS when it mentions TypeScript or `@langchain/langgraph`, is **not** part of Phase 1. In this phase the language is always chosen explicitly through the parameter.

### 8. Mini golden set — `eval/mini_golden.jsonl`

Fifteen lines, each with this shape:

```json
{"id": "q01", "question": "How do I add short-term memory to a graph?", "expected_slug": "langgraph/add-memory", "expected_section": "Add short-term memory", "language": "python", "verified": false}
```

All 15 questions are `"language": "python"`, so the strategy comparison is not affected by the language views. The eval script still reads the field and passes it to `search()`, so JS questions can be added later without code changes.

**Draft the questions:**

- Read the cleaned pages and draft 15 real developer questions that one page section clearly answers.
- Spread them across at least 10 pages: persistence, memory, interrupts, streaming, subgraphs, time travel, the Graph API vs the Functional API, errors such as `GRAPH_RECURSION_LIMIT`, and stores.
- Make 4 or more of them include an exact identifier, for example `add_conditional_edges`, `InMemorySaver`, `Command` or `interrupt()`. These are the cases dense retrieval tends to miss, and they motivate Phase 4.
- Do **not** copy section headings word for word into the question. Paraphrase the way a user would ask.

Set `"verified": false` on every question. Prateek checks each one and flips it to `true`.

### 9. Retrieval evaluation — `scripts/eval_retrieval.py`

```
python scripts/eval_retrieval.py --strategy both --k 5
```

**Scoring:**

- Each question is searched with its own `language` field.
- A result is a **hit** if its `page_slug == expected_slug` and `expected_section` appears case-insensitively in its `heading_path`. If `expected_section` is empty, a page match is enough.
- **Recall@5** is the share of questions with at least one hit in the top 5.
- **MRR** is the mean of `1/rank` of the first hit, or 0 if there is none.
- **Avg tokens returned** is the mean total tokens across the top-k results. This matters because parents are bigger: the comparison must show the context-size cost, not only recall.
- Also report Recall@1 and Recall@3.

**Output:**

- Print a warning listing any `verified: false` questions.
- Print a comparison table:

```
strategy       R@1   R@3   R@5   MRR    avg_tokens_returned
recursive      ...
parent_child   ...
```

- Also print a per-question table: hit rank per strategy, or `-` for a miss.
- Save `eval/results/phase1_<YYYYmmdd_HHMM>.json` with config, corpus commit, model, per-question ranks and the summary.

## Tests — `tests/`

**`test_cleaner.py`:**

- Python view: a `:::js` block is removed, and a `:::python` block keeps its code without the markers.
- JS view: the same input keeps the `:::js` code and removes the `:::python` block.
- Shared text outside any language block appears in both views, unchanged.
- Indented markers inside `<Tabs>` are handled in both views.
- `<Tab title="Postgres">` becomes `**Postgres**`, and the closing tags disappear.
- A snippet import plus `<XyzPy />` is replaced by the snippet code in the Python view and removed in the JS view; `<XyzJs />` does the opposite.
- `# [!code highlight]` is removed.
- Python code containing `x < y` and a `:::` string inside a fence is left unchanged.

**`test_chunker.py`:**

- A long page with a 700-token code block: the code block stays whole in exactly one chunk.
- Recursive chunks stay within the size limit, and consecutive chunks overlap.
- Parent-child: every child's `parent_id` exists, children stay within 200 tokens, parents within 1,500, and no child spans two parents.
- Chunk IDs are deterministic across two runs, and the Python and JS views of the same page produce no duplicate IDs.
- Every chunk carries the `language` of the view it came from.

**`test_retrieval.py`:**

- Parent de-duplication: given fake child hits `[p1, p1, p2, p3, p1, p4, p5, p6]`, the function returns `[p1, p2, p3, p4, p5]` in that order.
- Language filter: with an in-memory Chroma collection (`chromadb.EphemeralClient()`) holding Python and JS chunks with tiny fake embeddings, `search(..., language="python")` returns only Python chunks and `language="js"` only JS chunks, and each still returns `k` results when enough exist.

Tests must not download models. Inject a fake tokenizer that counts whitespace-separated words, through a parameter or a monkeypatch.

## Checkpoint — run these and report back

```bash
python scripts/download_docs.py
python scripts/ingest.py --strategy both --reset
python scripts/search.py "How do I add memory to a StateGraph?" --strategy both --k 3
python scripts/search.py "How do I add memory to a StateGraph?" --strategy parent_child --k 3 --language js
python scripts/eval_retrieval.py --strategy both --k 5
pytest -q
```

Report to Claude (chat):

1. The corpus commit hash, page count, and any views skipped per language.
2. Per strategy **and language**: chunk count (parents and children for parent-child), average tokens, number of oversized code chunks, and total ingest time with the device used.
3. The eval comparison table: R@1, R@3, R@5, MRR and avg tokens returned.
4. One question where retrieval clearly failed, with what came back instead.
5. Whether the JS search returned only TypeScript code, and whether anything in `data/clean/python` or `data/clean/js` still looks dirty.

## Out of scope for Phase 1 (do not build)

- LLM calls, prompts, answer generation (Phase 2)
- BM25, hybrid search, reranking (Phase 4)
- Qdrant (Phase 5)
- Automatic language detection from the question, and a UI language toggle (Phase 5 / 6)
- RAGAS or LLM-as-judge (Phase 3)
- FastAPI, Streamlit, Docker

## Code style

- Simple variable names, minimal comments, type hints on public functions.
- Small functions, no classes unless listed above.
- No frameworks beyond `requirements.txt`.
- All tunable numbers come from `config/settings.yaml`.
- `.gitignore` covers `data/`, `rag_env/`, `.env`, `__pycache__/` and `eval/results/*.json`; keep `eval/results/.gitkeep`.
- Commit after each numbered step. Tag `phase-1-done` when the checkpoint passes.
