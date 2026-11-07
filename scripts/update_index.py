import argparse
import io
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from download_docs import fetch, write_manifest

from documentor.config import collection_name, load_config, resolve
from documentor.evaluation import load_jsonl
from documentor.ingest import chunker
from documentor.ingest.loader import load_pages
from documentor.ingest.updater import (apply_update, load_manifest, manifest_from_index, plan_update, save_manifest,
                                       stale_golden)
from documentor.store.base import get_store
from documentor.store.parent_store import ParentStore

STRATEGIES = ["recursive", "parent_child"]


def git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def stats_commit() -> str:
    commits = {json.loads(resolve(f"data/ingest_stats_{s}.json").read_text())["commit"] for s in STRATEGIES}
    if len(commits) != 1:
        raise SystemExit(f"The ingest stats disagree on the corpus commit ({', '.join(commits)}). Run ingest.py --reset.")
    return commits.pop()


def set_stats_commit(commit: str) -> None:
    for s in STRATEGIES:
        path = resolve(f"data/ingest_stats_{s}.json")
        stats = json.loads(path.read_text())
        path.write_text(json.dumps(stats | {"commit": commit}, indent=2))


def stores_for(cfg: dict) -> tuple[dict, dict]:
    model = cfg["embedding"]["model"]
    stores = {s: get_store(cfg, collection_name(s, model)) for s in STRATEGIES}
    parents = {"parent_child": ParentStore(resolve(cfg["store"]["parents_path"]) / f"{collection_name('parent_child', model)}.json")}
    return stores, parents


def upstream_copy(cfg: dict, out: Path) -> str:
    """Fetch the latest commit and extract the tracked folders into `out`, leaving the checkout as it is."""
    corpus = cfg["corpus"]
    raw = resolve(corpus["raw_dir"])
    git(["fetch", "--depth", "1", "origin", corpus["branch"]], raw)
    commit = git(["rev-parse", "FETCH_HEAD"], raw)
    data = subprocess.run(["git", "archive", "FETCH_HEAD", *corpus["paths"]], cwd=raw, check=True,
                          capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="data")
    return commit


def init(cfg: dict) -> None:
    raw = resolve(cfg["corpus"]["raw_dir"])
    checkout, indexed = git(["rev-parse", "HEAD"], raw), stats_commit()
    if checkout != indexed:
        raise SystemExit(f"The docs checkout is at {checkout[:12]} but the index was built from {indexed[:12]}. "
                         "Run scripts/ingest.py --strategy both --reset instead.")
    pages = load_pages(cfg, cfg["corpus"]["languages"], count_tokens=chunker.count_tokens, write=False)
    stores, parents = stores_for(cfg)
    manifest, missing = manifest_from_index(pages, stores, parents, checkout)
    if missing:
        print(f"WARNING: {len(missing)} views have no chunks in some store: {', '.join(missing[:10])}")
    path = resolve(cfg["index"]["manifest_path"])
    save_manifest(path, manifest)
    print(f"Manifest for {len(manifest['pages'])} views at {checkout[:12]} written to {path.relative_to(resolve('.'))}. "
          "Nothing was embedded.")


def print_plan(plan) -> None:
    print("  ".join(f"{name} {n}" for name, n in plan.counts.items()))
    for name in ("added", "changed", "removed"):
        for key in getattr(plan, name):
            print(f"  {name:<8} {key}")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Re-embed only the doc pages that changed since the last ingest or update.")
    ap.add_argument("--init", action="store_true", help="write the manifest for the existing index, without embedding")
    ap.add_argument("--no-pull", action="store_true", help="use the docs checkout as it is")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    args = ap.parse_args()
    chunker.TOKENIZER_NAME = cfg["embedding"]["tokenizer"]
    if args.init:
        return init(cfg)

    start = time.perf_counter()
    path = resolve(cfg["index"]["manifest_path"])
    if not path.exists():
        raise SystemExit("No index manifest. Run scripts/update_index.py --init first.")
    manifest = load_manifest(path)
    raw = resolve(cfg["corpus"]["raw_dir"])
    with tempfile.TemporaryDirectory(dir=resolve("data")) as tmp:
        source = cfg
        if args.no_pull:
            commit = git(["rev-parse", "HEAD"], raw)
        elif args.dry_run:
            commit = upstream_copy(cfg, Path(tmp))
            source = cfg | {"corpus": cfg["corpus"] | {"raw_dir": tmp}}
        else:
            commit = fetch(cfg)
            write_manifest(cfg, commit)
        print(f"Docs at {commit[:12]} (index built from {manifest.get('corpus_commit', '?')[:12]})")
        pages = load_pages(source, cfg["corpus"]["languages"], count_tokens=chunker.count_tokens, write=False)
    plan = plan_update(pages, manifest)
    print_plan(plan)
    if args.dry_run:
        print("Dry run: nothing changed.")
        return

    stores, parents = stores_for(cfg)
    embedder = None

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal embedder
        if embedder is None:  # loaded only when something changed
            from documentor.retrieval.dense import get_embedder
            embedder = get_embedder(cfg["embedding"]["model"])
        return embedder.embed_documents(texts)

    new, summary = apply_update(plan, manifest, cfg, stores, parents, embed, commit, resolve(cfg["corpus"]["clean_dir"]))
    save_manifest(path, new)
    set_stats_commit(commit)

    golden = load_jsonl(resolve(cfg["evaluation"]["golden_path"]))
    views = plan.views
    read = lambda language, slug: (views.get(f"{language}:{slug}") or {}).get("text")
    stale = stale_golden(golden, read)
    print(f"Golden questions whose expected section is gone: {', '.join(stale) or 'none'}")
    print(f"chunks deleted {summary['chunks_deleted']:,} · added {summary['chunks_added']:,} · "
          f"parents deleted {summary['parents_deleted']:,} · added {summary['parents_added']:,} · "
          f"embedding calls {summary['embed_calls']} ({summary['embed_s']:.1f} s) · "
          f"BM25 rebuilt for {', '.join(summary['bm25_rebuilt']) or 'nothing'} · total {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
