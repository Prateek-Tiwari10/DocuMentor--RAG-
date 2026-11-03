import json


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def expected_entries(item: dict) -> list[dict]:
    """Golden expectations as a list; old one-answer lines become a one-entry list."""
    if "expected" in item:
        return item["expected"]
    return [{"slug": item["expected_slug"], "section": item.get("expected_section", "")}]


def text_headings(text: str) -> list[str]:
    """Markdown heading lines inside a chunk, skipping `#` comments in code fences."""
    headings, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            headings.append(line)
    return headings


def matches(result: dict, slug: str, section: str) -> bool:
    if result.get("page_slug") != slug:
        return False
    section = (section or "").lower()
    if not section or section in (result.get("heading_path") or "").lower():
        return True
    # A parent (or a recursive chunk spanning headings) can contain the expected subsection.
    return any(section in h.lower() for h in text_headings(result.get("text") or ""))


def is_hit(result: dict, item: dict) -> bool:
    return any(matches(result, e["slug"], e.get("section", "")) for e in expected_entries(item))


def first_hit_rank(results: list[dict], item: dict) -> int | None:
    return next((r["rank"] for r in results if is_hit(r, item)), None)
