import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SENTENCE_RE = re.compile(r"(?<=[.?!])\s+")

_counter: Callable[[str], int] | None = None
TOKENIZER_NAME = "BAAI/bge-m3"


@dataclass
class Block:
    kind: str                      # heading | code | text
    text: str
    level: int = 0
    heading_path: list[str] = field(default_factory=list)
    overlap: bool = False          # True for the tail carried over from the previous chunk


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tok.model_max_length = 10**9  # we only count, never feed the model here
    return tok


def set_token_counter(fn: Callable[[str], int] | None) -> None:
    """Override token counting (tests inject a whitespace counter)."""
    global _counter
    _counter = fn


def count_tokens(text: str) -> int:
    if _counter:
        return _counter(text)
    return len(_tokenizer().encode(text, add_special_tokens=False))


def parse_blocks(markdown: str, title: str = "") -> list[Block]:
    blocks, para, fence, code = [], [], None, []
    stack: dict[int, str] = {}

    def path():
        return ([title] if title else []) + [stack[k] for k in sorted(stack)]

    def flush():
        if para:
            blocks.append(Block("text", "\n".join(para).strip("\n"), heading_path=path()))
            para.clear()

    for line in markdown.splitlines():
        m = FENCE_RE.match(line)
        if fence:
            code.append(line)
            if m and m.group(1).startswith(fence) and not m.group(2).strip():
                blocks.append(Block("code", "\n".join(code), heading_path=path()))
                fence, code = None, []
            continue
        if m:
            flush()
            fence, code = m.group(1), [line]
            continue
        h = HEADING_RE.match(line)
        if h:
            flush()
            level = len(h.group(1))
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = h.group(2)
            blocks.append(Block("heading", line, level=level, heading_path=path()))
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    if code:  # unclosed fence: keep it whole anyway
        blocks.append(Block("code", "\n".join(code), heading_path=path()))
    return blocks


def _join(blocks: list[Block]) -> str:
    return "\n\n".join(b.text for b in blocks)


def _merge(pieces: list[str], sep: str, max_tokens: int) -> list[str]:
    out, cur = [], ""
    for piece in pieces:
        cand = f"{cur}{sep}{piece}" if cur else piece
        if cur and count_tokens(cand) > max_tokens:
            out.append(cur)
            cur = piece
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_tokens: int, level: int = 0) -> list[str]:
    """Split one oversized text block: lines, then sentences, then words."""
    if count_tokens(text) <= max_tokens:
        return [text]
    splitters = [("\n", lambda t: t.split("\n")), (" ", SENTENCE_RE.split), (" ", lambda t: t.split(" "))]
    for i in range(level, len(splitters)):
        sep, fn = splitters[i]
        pieces = [p for p in fn(text) if p.strip()]
        if len(pieces) > 1:
            out = []
            for piece in _merge(pieces, sep, max_tokens):
                out += split_text(piece, max_tokens, i + 1) if count_tokens(piece) > max_tokens else [piece]
            return out
    return [text]


def _tail(block: Block, overlap: int) -> Block | None:
    """Whole sentences from the end of a text block, about `overlap` tokens."""
    if block.kind != "text" or overlap <= 0:
        return None
    sentences = [s for s in SENTENCE_RE.split(block.text.replace("\n", " ")) if s.strip()]
    tail = []
    for s in reversed(sentences):
        if tail and count_tokens(" ".join([s] + tail)) > overlap:
            break
        tail.insert(0, s)
    text = " ".join(tail)
    if count_tokens(text) > overlap:  # one long sentence: fall back to its last words
        words = text.split(" ")
        while len(words) > 1 and count_tokens(" ".join(words)) > overlap:
            words = words[len(words) // 4 or 1:]
        text = " ".join(words)
    return Block("text", text, heading_path=block.heading_path, overlap=True)


def _expand(blocks: list[Block], size: int) -> list[Block]:
    out = []
    for b in blocks:
        if b.kind == "text" and count_tokens(b.text) > size:
            out += [Block("text", t, heading_path=b.heading_path) for t in split_text(b.text, size)]
        else:
            out.append(b)
    return out


def pack(blocks: list[Block], size: int, overlap: int,
         heading_break: float | None = None) -> list[dict]:
    """Greedily pack blocks into chunks of at most `size` tokens, carrying an overlap tail."""
    queue = deque(_expand(blocks, size))
    chunks, cur = [], []

    def has_body(bs):
        return any(not x.overlap and x.kind != "heading" for x in bs)

    def emit(bs):
        text = _join(bs)
        first = next(x for x in bs if not x.overlap)
        tokens = count_tokens(text)
        chunks.append({"text": text, "heading_path": first.heading_path, "tokens": tokens,
                       "oversized": tokens > size, "blocks": [x for x in bs if not x.overlap]})

    def close(carry: bool) -> list[Block]:
        trailing = []
        while cur and cur[-1].kind == "heading":  # a chunk never ends on a heading
            trailing.insert(0, cur.pop())
        emit(cur)
        tail = _tail(cur[-1], overlap) if carry else None
        return ([tail] if tail else []) + trailing

    while queue:
        b = queue.popleft()
        if b.kind == "code" and count_tokens(b.text) > size:
            heads = close(carry=False) if has_body(cur) else [x for x in cur if not x.overlap]
            emit(heads + [b])
            cur = []
            continue
        tokens = count_tokens(_join(cur + [b]))
        early = (heading_break is not None and b.kind == "heading" and b.level in (2, 3)
                 and tokens > heading_break * size)
        if has_body(cur) and (tokens > size or early):
            cur = close(carry=True)
            tokens = count_tokens(_join(cur + [b]))
        if tokens > size and any(x.overlap for x in cur):
            cur = [x for x in cur if not x.overlap]
            tokens = count_tokens(_join(cur + [b]))
        if tokens > size and b.kind == "text" and cur:  # only headings left in cur
            room = size - count_tokens(_join(cur)) - 2
            parts = split_text(b.text, room) if room > 0 else []
            if len(parts) > 1:
                queue.extendleft(Block("text", t, heading_path=b.heading_path) for t in reversed(parts[1:]))
                b = Block("text", parts[0], heading_path=b.heading_path)
        cur.append(b)
    if has_body(cur):
        close(carry=False)
    return chunks


def _id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _embed_text(text: str, heading_path: list[str], prepend: bool) -> str:
    return f"{' > '.join(heading_path)}\n\n{text}" if prepend else text


def _record(page: dict, chunk: dict, strategy: str, index: int, parent_id: str | None,
            prepend: bool) -> dict:
    return {
        "id": _id(strategy, page["language"], page["slug"], index),
        "text": chunk["text"],
        "embed_text": _embed_text(chunk["text"], chunk["heading_path"], prepend),
        "page_slug": page["slug"],
        "language": page["language"],
        "url": page["url"],
        "title": page["title"],
        "heading_path": chunk["heading_path"],
        "tokens": chunk["tokens"],
        "strategy": strategy,
        "parent_id": parent_id,
        "oversized": chunk.get("oversized", False),
    }


def chunk_recursive(page: dict, cfg: dict) -> list[dict]:
    c = cfg["chunking"]
    blocks = parse_blocks(page["text"], page["title"])
    chunks = pack(blocks, c["recursive"]["size"], c["recursive"]["overlap"],
                  heading_break=c.get("heading_break_ratio", 0.6))
    return [_record(page, ch, "recursive", i, None, c["prepend_headings"]) for i, ch in enumerate(chunks)]


def _sections(blocks: list[Block], level: int) -> list[list[Block]]:
    """Split blocks at headings of `level` (or higher); content before the first one is its own section."""
    sections = [[]]
    for b in blocks:
        if b.kind == "heading" and b.level <= level and sections[-1]:
            sections.append([])
        sections[-1].append(b)
    return [s for s in sections if s]


def _parent_pieces(blocks: list[Block], parent_max: int) -> list[dict]:
    text = _join(blocks)
    if count_tokens(text) <= parent_max:
        return [{"blocks": blocks, "heading_path": blocks[0].heading_path}]
    pieces = []
    for sub in _sections(blocks, 3):
        if count_tokens(_join(sub)) <= parent_max:
            pieces.append({"blocks": sub, "heading_path": sub[0].heading_path})
            continue
        for ch in pack(sub, parent_max, 0):
            pieces.append({"blocks": ch["blocks"], "heading_path": ch["heading_path"],
                           "oversized": ch["oversized"]})
    return pieces


def chunk_parent_child(page: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    c = cfg["chunking"]
    pc = c["parent_child"]
    blocks = parse_blocks(page["text"], page["title"])
    parents, children = [], []
    for section in _sections(blocks, 2):
        for piece in _parent_pieces(section, pc["parent_max"]):
            text = _join(piece["blocks"])
            parent_id = _id("parent", page["language"], page["slug"], len(parents))
            parents.append({
                "parent_id": parent_id,
                "text": text,
                "page_slug": page["slug"],
                "language": page["language"],
                "url": page["url"],
                "title": page["title"],
                "heading_path": piece["heading_path"],
                "tokens": count_tokens(text),
                "oversized": piece.get("oversized", False),
            })
            for ch in pack(piece["blocks"], pc["child_size"], pc["child_overlap"]):
                children.append(_record(page, ch, "parent_child", len(children), parent_id,
                                        c["prepend_headings"]))
    return children, parents
