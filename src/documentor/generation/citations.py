import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
FENCE = re.compile(r"^\s*```", re.M)
INLINE_CODE = re.compile(r"(`[^`\n]+`)")


@dataclass
class Parsed:
    text: str            # answer with invalid citations removed
    refused: bool
    cited: list[int]     # valid block numbers, in order of first appearance
    invalid: list[int]   # numbers removed because they weren't in the context
    uncited: bool        # answered, but no valid citation anywhere


def is_refusal(text: str, sentinel: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(sentinel) and not re.sub(r"[\s\W]", "", stripped[len(sentinel):])


def split_fences(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, is_code) pieces on ``` fence lines."""
    pieces, pos, in_fence = [], 0, False
    for m in FENCE.finditer(text):
        if in_fence:
            end = text.find("\n", m.end())
            end = len(text) if end == -1 else end + 1
            pieces.append((text[pos:end], True))
            pos = end
        else:
            pieces.append((text[pos:m.start()], False))
            pos = m.start()
        in_fence = not in_fence
    pieces.append((text[pos:], in_fence))
    return pieces


def parse_answer(text: str, blocks: list[dict], sentinel: str) -> Parsed:
    if is_refusal(text, sentinel):
        return Parsed(text=text.strip(), refused=True, cited=[], invalid=[], uncited=False)
    if sentinel in text:
        log.warning("refusal sentinel inside a longer answer; treating it as answered")

    valid = set(range(1, len(blocks) + 1))
    cited: list[int] = []
    invalid: list[int] = []

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        keep = [n for n in nums if n in valid]
        invalid.extend(n for n in nums if n not in valid)
        cited.extend(n for n in keep if n not in cited)
        return f"[{', '.join(map(str, keep))}]" if keep else ""

    def outside_inline(seg: str) -> str:
        parts = INLINE_CODE.split(seg)
        return "".join(p if i % 2 else CITATION.sub(fix, p) for i, p in enumerate(parts))

    out = "".join(seg if code else outside_inline(seg) for seg, code in split_fences(text))
    if invalid:
        out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
    return Parsed(text=out.strip(), refused=False, cited=cited, invalid=invalid, uncited=not cited)


def sources_for(parsed: Parsed, blocks: list[dict]) -> list[dict]:
    by_n = {b["n"]: b for b in blocks}
    return [{k: by_n[n][k] for k in ("n", "url", "heading_path", "page_slug")} for n in parsed.cited]
