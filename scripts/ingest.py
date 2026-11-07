import argparse
import copy
import json
import logging
import statistics
import time

from documentor.config import collection_name, load_config, resolve
from documentor.embed.embedder import Embedder
from documentor.ingest import chunker
from documentor.ingest.loader import load_pages
from documentor.ingest.updater import manifest_from_index, save_manifest
from documentor.language import build_language_vocab
from documentor.retrieval.bm25 import build_and_save
from documentor.store.base import get_store
from documentor.store.parent_store import ParentStore

STRATEGIES = ["recursive", "parent_child"]


def parse_args(cfg: dict) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean, chunk, embed and store the LangGraph docs.")
    ap.add_argument("--strategy", choices=STRATEGIES + ["both"], default=cfg["chunking"]["strategy"])
    ap.add_argument("--languages", default=",".join(cfg["corpus"]["languages"]))
    ap.add_argument("--fast", action="store_true", help="debug with all-MiniLM-L6-v2")
    ap.add_argument("--reset", action="store_true", help="drop the collection before ingesting")
    return ap.parse_args()


def avg(values: list[int]) -> int:
    return round(statistics.mean(values)) if values else 0


def language_stats(strategy: str, language: str, pages: list[dict], chunks: list[dict],
                   parents: list[dict]) -> dict:
    chunks = [c for c in chunks if c["language"] == language]
    parents = [p for p in parents if p["language"] == language]
    stats = {"strategy": strategy, "language": language,
             "pages": sum(p["language"] == language for p in pages)}
    if strategy == "recursive":
        stats |= {"chunks": len(chunks), "avg_chunk_tokens": avg([c["tokens"] for c in chunks])}
    else:
        stats |= {"parents": len(parents), "children": len(chunks),
                  "avg_child_tokens": avg([c["tokens"] for c in chunks]),
                  "avg_parent_tokens": avg([p["tokens"] for p in parents]),
                  "oversized_parents": sum(p["oversized"] for p in parents)}
    stats["oversized_code"] = sum(c["oversized"] for c in chunks)
    return stats


def print_stats(stats: dict) -> None:
    print("  ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in stats.items()))


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = copy.deepcopy(load_config())
    args = parse_args(cfg)
    emb_cfg = cfg["embedding"]
    model = emb_cfg["fast_model"] if args.fast else emb_cfg["model"]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    strategies = STRATEGIES if args.strategy == "both" else [args.strategy]
    manifest = json.loads(resolve("data/raw/manifest.json").read_text())
    chunker.TOKENIZER_NAME = emb_cfg["tokenizer"]

    start = time.time()
    pages = load_pages(cfg, languages, count_tokens=chunker.count_tokens)
    print(f"Loaded {len(pages)} page views for {languages} at commit {manifest['commit'][:12]}")
    embedder = Embedder(model, emb_cfg["device"], emb_cfg["batch_size"])

    for strategy in strategies:
        t0 = time.time()
        chunks, parents = chunker.chunk_pages(pages, strategy, cfg)
        name = collection_name(strategy, model)
        store = get_store(cfg, name)
        if args.reset:
            store.reset()
        print(f"[{strategy}] embedding {len(chunks):,} chunks into '{name}'")
        embeddings = embedder.embed_documents([c["embed_text"] for c in chunks])
        store.add(chunks, embeddings)
        if parents:
            ParentStore(resolve(cfg["store"]["parents_path"]) / f"{name}.json").save(parents, merge=not args.reset)
        for language in languages:  # so a re-ingest never leaves a stale keyword index
            index, seconds = build_and_save(store, language, cfg, manifest["commit"])
            print(f"bm25 {language}: {index.meta['chunk_count']:,} chunks, {index.vocabulary_size:,} terms, {seconds:.2f}s")

        per_language = [language_stats(strategy, lang, pages, chunks, parents) for lang in languages]
        for stats in per_language:
            print_stats(stats)
        elapsed = round(time.time() - t0)
        print(f"strategy_time={elapsed}s  collection_count={store.count():,}")
        out = {"commit": manifest["commit"], "model": model, "device": embedder.device,
               "collection": name, "time_s": elapsed, "languages": per_language}
        resolve(f"data/ingest_stats_{strategy}.json").write_text(json.dumps(out, indent=2))
    if not args.fast:  # the manifest and language lists describe the main bge-m3 index
        stores = {s: get_store(cfg, collection_name(s, model)) for s in STRATEGIES}
        parent_path = resolve(cfg["store"]["parents_path"]) / f"{collection_name('parent_child', model)}.json"
        index_manifest, _ = manifest_from_index(pages, stores, {"parent_child": ParentStore(parent_path)}, manifest["commit"])
        save_manifest(resolve(cfg["index"]["manifest_path"]), index_manifest)
        vocab = build_language_vocab(cfg)
        print(f"manifest: {len(index_manifest['pages'])} views  language lists: "
              f"{len(vocab['python_only']):,} python-only, {len(vocab['js_only']):,} js-only")
    print(f"total_time={round(time.time() - start)}s  device={embedder.device}")


if __name__ == "__main__":
    main()
