import copy

import pytest

from documentor.config import load_config
from documentor.ingest import chunker
from documentor.ingest.chunker import chunk_parent_child, chunk_recursive, parse_blocks


@pytest.fixture(autouse=True)
def word_tokenizer():
    chunker.set_token_counter(lambda text: len(text.split()))
    yield
    chunker.set_token_counter(None)


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


def sentence(i: int) -> str:
    return f"Sentence number {i} explains how a LangGraph checkpointer saves state."


def paragraph(start: int, n: int = 6) -> str:
    return " ".join(sentence(i) for i in range(start, start + n))


def long_page(language: str = "python") -> dict:
    code = "```python\n" + "\n".join(f"x_{i} = compute({i}) + other_value" for i in range(175)) + "\n```"
    parts = ["Intro paragraph about memory. " + paragraph(0)]
    for s in range(4):
        parts.append(f"## Section {s}")
        for p in range(6):
            parts.append(paragraph(100 * s + 10 * p))
        parts.append(f"### Subsection {s}")
        parts.append(paragraph(100 * s + 90))
        if s == 1:
            parts.append(code)
    return {"slug": "langgraph/test-page", "language": language, "url": "https://x/test-page",
            "title": "Test Page", "text": "\n\n".join(parts) + "\n"}


def fence_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("```"))


def test_parse_blocks_keeps_code_whole_and_tracks_headings():
    md = "Intro.\n\n## A\n\nText a.\n\n```python\nx = 1\n\ny = 2\n```\n\n### B\n\nText b.\n"
    blocks = parse_blocks(md, "Title")
    assert [b.kind for b in blocks] == ["text", "heading", "text", "code", "heading", "text"]
    assert blocks[3].text == "```python\nx = 1\n\ny = 2\n```"
    assert blocks[-1].heading_path == ["Title", "A", "B"]


def test_big_code_block_stays_whole(cfg):
    page = long_page()
    code_lines = [l for l in page["text"].splitlines() if l.startswith("x_")]
    assert len(" ".join(code_lines).split()) > 700
    for chunks in (chunk_recursive(page, cfg), chunk_parent_child(page, cfg)[0]):
        holders = [c for c in chunks if "x_0 = compute(0)" in c["text"]]
        assert len(holders) == 1
        assert all(line in holders[0]["text"] for line in code_lines)
        assert holders[0]["oversized"]
        assert all(fence_lines(c["text"]) % 2 == 0 for c in chunks)


def test_recursive_size_and_overlap(cfg):
    chunks = chunk_recursive(long_page(), cfg)
    assert len(chunks) > 3
    for c in chunks:
        assert c["oversized"] or c["tokens"] <= cfg["chunking"]["recursive"]["size"]
    overlapping = 0
    for a, b in zip(chunks, chunks[1:]):
        if a["oversized"] or b["oversized"] or a["text"].rstrip().endswith("```"):
            continue
        first = " ".join(b["text"].split("\n\n")[0].split())
        assert first in " ".join(a["text"].split())
        overlapping += 1
    assert overlapping >= 2


def test_parent_child_structure(cfg):
    pc = cfg["chunking"]["parent_child"]
    children, parents = chunk_parent_child(long_page(), cfg)
    by_id = {p["parent_id"]: p for p in parents}
    assert len(by_id) == len(parents)
    for c in children:
        assert c["parent_id"] in by_id
        assert c["oversized"] or c["tokens"] <= pc["child_size"]
        parent = by_id[c["parent_id"]]
        assert c["language"] == parent["language"]
        for piece in c["text"].split("\n\n"):  # no child spans two parents
            assert " ".join(piece.split()) in " ".join(parent["text"].split())
    for p in parents:
        assert p["oversized"] or p["tokens"] <= pc["parent_max"]
    assert parents[0]["heading_path"] == ["Test Page"]
    assert any(p["heading_path"][-1] == "Section 2" for p in parents)


def test_parent_split_at_h3_when_too_big(cfg):
    cfg["chunking"]["parent_child"]["parent_max"] = 400
    _, parents = chunk_parent_child(long_page(), cfg)
    assert any(p["heading_path"][-1] == "Subsection 0" for p in parents)
    assert all(p["oversized"] or p["tokens"] <= 400 for p in parents)


def test_ids_deterministic_and_unique_across_languages(cfg):
    py, js = long_page("python"), long_page("js")
    r1, r2 = chunk_recursive(py, cfg), chunk_recursive(py, cfg)
    assert [c["id"] for c in r1] == [c["id"] for c in r2]
    c1, p1 = chunk_parent_child(py, cfg)
    c2, p2 = chunk_parent_child(py, cfg)
    assert [c["id"] for c in c1] == [c["id"] for c in c2]
    assert [p["parent_id"] for p in p1] == [p["parent_id"] for p in p2]
    cj, pj = chunk_parent_child(js, cfg)
    ids = [c["id"] for c in r1 + chunk_recursive(js, cfg) + c1 + cj]
    assert len(ids) == len(set(ids))
    pids = [p["parent_id"] for p in p1 + pj]
    assert len(pids) == len(set(pids))


def test_chunks_carry_view_language(cfg):
    for language in ("python", "js"):
        page = long_page(language)
        children, parents = chunk_parent_child(page, cfg)
        for item in chunk_recursive(page, cfg) + children + parents:
            assert item["language"] == language


def test_embed_text_prefixes_heading_path(cfg):
    chunks = chunk_recursive(long_page(), cfg)
    c = chunks[-1]
    assert c["embed_text"].startswith(" > ".join(c["heading_path"]) + "\n\n")
    cfg["chunking"]["prepend_headings"] = False
    assert chunk_recursive(long_page(), cfg)[-1]["embed_text"] == c["text"]
