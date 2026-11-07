import argparse
import json

from documentor.config import load_config, resolve
from documentor.ingest import chunker
from documentor.ingest.chunker import make_embed_text
from documentor.ingest.loader import load_pages
from documentor.retrieval.bm25 import build_and_save, index_path
from documentor.retrieval.dense import get_store

STRATEGIES = ["recursive", "parent_child"]


def sanity_check(store, strategy: str, page: dict, cfg: dict) -> int:
    """Re-chunk one page: the embed_text rebuilt from Chroma must match the chunker's, id for id."""
    if strategy == "recursive":
        chunks = chunker.chunk_recursive(page, cfg)
    else:
        chunks = chunker.chunk_parent_child(page, cfg)[0]
    stored = {c["id"]: c for c in store.by_ids([c["id"] for c in chunks])}
    prepend = cfg["chunking"]["prepend_headings"]
    for chunk in chunks:
        row = stored.get(chunk["id"])
        if row is None:
            raise SystemExit(f"Sanity check failed: chunk {chunk['id']} of {page['slug']} is not in {store.name}. Re-ingest.")
        if make_embed_text(row["title"], row["heading_path"], row["text"], prepend) != chunk["embed_text"]:
            raise SystemExit(f"Sanity check failed: embed_text differs for chunk {chunk['id']} of {page['slug']}.")
    return len(chunks)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Build the BM25 indexes from the chunks stored in Chroma.")
    ap.add_argument("--strategy", choices=STRATEGIES + ["both"], default="both")
    ap.add_argument("--languages", default=",".join(cfg["corpus"]["languages"]))
    args = ap.parse_args()
    strategies = STRATEGIES if args.strategy == "both" else [args.strategy]
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    model = cfg["embedding"]["model"]
    commit = json.loads(resolve("data/raw/manifest.json").read_text())["commit"]
    chunker.TOKENIZER_NAME = cfg["embedding"]["tokenizer"]
    pages = {lang: load_pages(cfg, [lang], count_tokens=chunker.count_tokens)[0] for lang in languages}

    print(f"{'collection':<30} {'language':<8} {'chunks':>7} {'vocabulary':>10} {'build s':>8}  sanity")
    for strategy in strategies:
        store = get_store(strategy, model)
        for language in languages:
            checked = sanity_check(store, strategy, pages[language], cfg)
            index, seconds = build_and_save(store, language, cfg, commit)
            print(f"{store.name:<30} {language:<8} {index.meta['chunk_count']:>7,} {index.vocabulary_size:>10,} "
                  f"{seconds:>8.2f}  {checked} chunks of {pages[language]['slug']} match")
    print(f"\nIndexes in {index_path(cfg, 'x', 'y').parent.relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
