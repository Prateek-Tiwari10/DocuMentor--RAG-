import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from documentor.evaluation.golden import section_exists
from documentor.ingest.chunker import chunk_pages
from documentor.ingest.loader import write_view
from documentor.language import build_language_vocab
from documentor.retrieval.bm25 import build_and_save

CLASSES = ("added", "changed", "removed", "unchanged")


def view_key(language: str, slug: str) -> str:
    return f"{language}:{slug}"


def split_key(key: str) -> tuple[str, str]:
    language, slug = key.split(":", 1)
    return language, slug


def view_hash(text: str) -> str:
    """Hash of the cleaned view, so an edited snippet changes every page that imports it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class UpdatePlan:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    views: dict[str, dict] = field(default_factory=dict)   # key -> page view, for every view that exists now

    @property
    def counts(self) -> dict[str, int]:
        return {c: len(getattr(self, c)) for c in CLASSES}

    @property
    def to_embed(self) -> list[str]:
        return self.added + self.changed

    @property
    def to_delete(self) -> list[str]:
        return self.changed + self.removed


def plan_update(pages: list[dict], manifest: dict) -> UpdatePlan:
    old = manifest.get("pages", {})
    plan = UpdatePlan(views={view_key(p["language"], p["slug"]): p for p in pages})
    for key, page in sorted(plan.views.items()):
        if key not in old:
            plan.added.append(key)
        elif old[key]["hash"] != view_hash(page["text"]):
            plan.changed.append(key)
        else:
            plan.unchanged.append(key)
    plan.removed = sorted(k for k in old if k not in plan.views)
    return plan


def page_entry(page: dict, chunk_counts: dict[str, int], parents: int) -> dict:
    return {"hash": view_hash(page["text"]), "chunks": chunk_counts, "parents": parents}


def new_manifest(entries: dict[str, dict], commit: str) -> dict:
    return {"corpus_commit": commit, "updated_at": datetime.now().isoformat(timespec="seconds"),
            "pages": dict(sorted(entries.items()))}


def apply_update(plan: UpdatePlan, manifest: dict, cfg: dict, stores: dict, parent_stores: dict,
                 embed: Callable[[list[str]], list[list[float]]], commit: str, clean_dir: Path,
                 vocab_out: Path | None = None) -> tuple[dict, dict]:
    """Delete changed and removed pages, re-add changed and added ones, then rebuild BM25 and the language lists.
    `stores` and `parent_stores` map strategy -> store; `embed` is only called when something needs embedding."""
    start = time.perf_counter()
    summary = {"chunks_deleted": 0, "chunks_added": 0, "parents_deleted": 0, "parents_added": 0,
               "embed_calls": 0, "embed_s": 0.0}
    entries = {k: v for k, v in manifest.get("pages", {}).items() if k not in plan.to_delete}

    for key in plan.to_delete:
        language, slug = split_key(key)
        for strategy, store in stores.items():
            summary["chunks_deleted"] += store.delete_page(slug, language)
            if strategy in parent_stores:
                summary["parents_deleted"] += parent_stores[strategy].remove_page(slug, language)
    for key in plan.removed:
        language, slug = split_key(key)
        (Path(clean_dir) / language / f"{slug}.md").unlink(missing_ok=True)

    pages = [plan.views[k] for k in plan.to_embed]
    counts = {k: {} for k in plan.to_embed}
    parent_counts = {k: 0 for k in plan.to_embed}
    for page in pages:
        write_view(Path(clean_dir), page["language"], page["slug"], page["text"])
    for strategy, store in stores.items():
        chunks, parents = chunk_pages(pages, strategy, cfg)
        if chunks:
            t = time.perf_counter()
            embeddings = embed([c["embed_text"] for c in chunks])
            summary["embed_calls"] += 1
            summary["embed_s"] += time.perf_counter() - t
            store.add(chunks, embeddings)
            summary["chunks_added"] += len(chunks)
        if parents and strategy in parent_stores:
            parent_stores[strategy].save(parents, merge=True)
            summary["parents_added"] += len(parents)
        for chunk in chunks:
            key = view_key(chunk["language"], chunk["page_slug"])
            counts[key][strategy] = counts[key].get(strategy, 0) + 1
        for parent in parents:
            parent_counts[view_key(parent["language"], parent["page_slug"])] += 1
    for key in plan.to_embed:
        entries[key] = page_entry(plan.views[key], {s: counts[key].get(s, 0) for s in stores}, parent_counts[key])

    touched = sorted({split_key(k)[0] for k in plan.to_embed + plan.removed})
    for store in stores.values():
        for language in touched:
            build_and_save(store, language, cfg, commit)
    if touched:
        build_language_vocab(cfg, Path(clean_dir), vocab_out)
    summary |= {"bm25_rebuilt": touched, "total_s": time.perf_counter() - start}
    return new_manifest(entries, commit), summary


def stale_golden(golden: list[dict], read) -> list[str]:
    """Golden questions whose expected section is gone from its page. Reported only, never edited."""
    return [q["id"] for q in golden if not section_exists(q, read)]


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {"pages": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1))


def manifest_from_index(pages: list[dict], stores: dict, parent_stores: dict, commit: str) -> tuple[dict, list[str]]:
    """A manifest for what the index holds now, counted from the stores, so nothing is re-embedded.
    Also returns the views that have no chunks in some store, which means the index and the docs disagree."""
    per_page = {s: {} for s in stores}
    for strategy, store in stores.items():
        for language in sorted({p["language"] for p in pages}):
            for chunk in store.get_all(language):
                key = view_key(language, chunk["page_slug"])
                per_page[strategy][key] = per_page[strategy].get(key, 0) + 1
    entries, missing = {}, []
    for page in pages:
        key = view_key(page["language"], page["slug"])
        counts = {s: per_page[s].get(key, 0) for s in stores}
        if not all(counts.values()):
            missing.append(key)
        parents = sum(ps.count_page(page["slug"], page["language"]) for ps in parent_stores.values())
        entries[key] = page_entry(page, counts, parents)
    return new_manifest(entries, commit), missing
