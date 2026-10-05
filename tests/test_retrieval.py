import uuid

import chromadb

from documentor.retrieval.dense import group_by_parent, search
from documentor.store.chroma_store import ChromaStore
from documentor.store.parent_store import ParentStore

DIM = 4


class FakeEmbedder:
    """Every text maps to the same direction plus a little noise, so all chunks are similar."""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def vec(i: int) -> list[float]:
    return [1.0, 0.01 * i, 0.0, 0.0]


def make_store() -> ChromaStore:
    return ChromaStore(None, f"test_{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())


def chunk(i: int, language: str, parent_id: str | None = None) -> dict:
    return {"id": f"{language}-{i}", "text": f"{language} chunk {i}", "embed_text": "x",
            "page_slug": f"langgraph/page-{i % 3}", "language": language, "url": "u", "title": "T",
            "heading_path": ["T", f"Section {i}"], "tokens": 10, "strategy": "s", "parent_id": parent_id}


def test_parent_dedup_keeps_rank_order():
    hits = [{"parent_id": p, "score": 1 - i * 0.01, "text": f"c{i}"}
            for i, p in enumerate(["p1", "p1", "p2", "p3", "p1", "p4", "p5", "p6"])]
    groups = group_by_parent(hits, k=5)
    assert [g["parent_id"] for g in groups] == ["p1", "p2", "p3", "p4", "p5"]
    assert groups[0]["matched_children"] == 3
    assert groups[0]["best_child_text"] == "c0"


def test_language_filter_recursive():
    store = make_store()
    chunks = [chunk(i, "python") for i in range(8)] + [chunk(i, "js") for i in range(8)]
    store.add(chunks, [vec(i) for i in range(8)] * 2)
    for language in ("python", "js"):
        results = search("q", "recursive", k=5, language=language, embedder=FakeEmbedder(), store=store)
        assert len(results) == 5
        assert all(r["language"] == language for r in results)
        assert all(r["text"].startswith(language) for r in results)
        assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]


def test_language_filter_parent_child(tmp_path):
    store = make_store()
    parents = ParentStore(tmp_path / "parents.json")
    parent_rows, children = [], []
    for language in ("python", "js"):
        for p in range(6):
            pid = f"{language}-p{p}"
            parent_rows.append({"parent_id": pid, "text": f"{language} parent {p}", "page_slug": "langgraph/x",
                                "language": language, "url": "u", "title": "T",
                                "heading_path": ["T", f"Section {p}"], "tokens": 100})
            children += [chunk(10 * p + c, language, pid) for c in range(2)]
    parents.save(parent_rows)
    store.add(children, [vec(i) for i in range(len(children))])
    for language in ("python", "js"):
        results = search("q", "parent_child", k=5, language=language, embedder=FakeEmbedder(),
                         store=store, parents=parents)
        assert len(results) == 5
        assert all(r["language"] == language for r in results)
        assert len({r["parent_id"] for r in results}) == 5
        assert all(r["heading_path"].startswith("T > ") for r in results)
