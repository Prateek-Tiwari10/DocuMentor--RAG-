import builtins
import keyword
import re
from pathlib import Path

from documentor.evaluation import expected_entries, is_hit, load_jsonl, text_headings

ALLOWED = set(dir(builtins)) | set(keyword.kwlist)
INLINE = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DEFINED = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
IMPORT = re.compile(r"^\s*(?:from\s+.+\s+import\s+.+|import\s+.+)$", re.M)


def load_golden(path) -> list[dict]:
    items = load_jsonl(path)
    for item in items:
        if not expected_entries(item):
            raise ValueError(f"{item.get('id')}: no expected entry")
    return items


def section_text(page: str, section: str) -> str:
    """The expected section and its subsections; a blank section means the whole page."""
    if not section:
        return page
    lines = page.split("\n")
    in_fence, start, level = False, None, 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            if start is None and section.lower() in line.lower():
                start, level = i, depth
            elif start is not None and depth <= level:
                return "\n".join(lines[start:i])
    return "\n".join(lines[start:]) if start is not None else ""


def read_page(clean_dir: Path, language: str, slug: str) -> str | None:
    path = Path(clean_dir) / language / f"{slug}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def expected_text(item: dict, read) -> str:
    """Text of every expected section. `read` maps (language, slug) to page text."""
    parts = []
    for entry in expected_entries(item):
        page = read(item.get("language", "python"), entry["slug"]) or ""
        parts.append(section_text(page, entry.get("section", "")))
    return "\n\n".join(p for p in parts if p)


def section_exists(item: dict, read) -> bool:
    """Every expected entry must hit a fake result built from its cleaned page."""
    for entry in expected_entries(item):
        page = read(item.get("language", "python"), entry["slug"])
        if not page:
            return False
        fake = {"page_slug": entry["slug"], "heading_path": "", "text": page}
        if not is_hit(fake, {"expected": [entry]}):
            return False
        if entry.get("section") and not any(entry["section"].lower() in h.lower() for h in text_headings(page)):
            return False
    return True


def code_identifiers(text: str) -> set[str]:
    """Names a reference commits to: backticked names, import lines, and function calls."""
    code = "\n".join(FENCE.findall(text))
    prose = FENCE.sub("", text)
    names = set()
    for span in INLINE.findall(prose):
        names |= set(NAME.findall(span))
    for line in IMPORT.findall(code):
        names |= set(NAME.findall(line)) - {"from", "import", "as"}
    names |= set(CALL.findall(code))
    return names - set(DEFINED.findall(code)) - ALLOWED


def missing_identifiers(reference: str, section: str) -> list[str]:
    return sorted(n for n in code_identifiers(reference) if not re.search(rf"\b{re.escape(n)}\b", section))


def auto_check(item: dict, reference: str, read) -> list[str]:
    """Problems with a question or its reference; an empty list means the checks passed."""
    if not section_exists(item, read):
        return ["expected section not found in the cleaned page"]
    problems = []
    if not reference.strip():
        problems.append("empty reference")
    if len(FENCE.findall(reference)) > 1:
        problems.append("more than one code snippet")
    missing = missing_identifiers(reference, expected_text(item, read))
    if missing:
        problems.append(f"names not in the section: {', '.join(missing)}")
    return problems
