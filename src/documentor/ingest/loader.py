import logging
from pathlib import Path
from typing import Callable

from documentor.config import resolve
from documentor.ingest.cleaner import clean_page

log = logging.getLogger(__name__)


def load_snippets(cfg: dict) -> dict[str, str]:
    snippet_dir = resolve(cfg["corpus"]["raw_dir"]) / cfg["corpus"]["snippet_dir"]
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(snippet_dir.rglob("*.mdx"))}


def page_paths(cfg: dict) -> list[Path]:
    raw_dir = resolve(cfg["corpus"]["raw_dir"])
    paths = []
    for page_dir in cfg["corpus"]["page_dirs"]:
        paths += [p for p in sorted((raw_dir / page_dir).rglob("*")) if p.suffix in (".mdx", ".md")]
    return paths


def page_slug(path: Path, cfg: dict) -> str:
    oss_dir = resolve(cfg["corpus"]["raw_dir"]) / "src" / "oss"
    return path.relative_to(oss_dir).with_suffix("").as_posix()


def load_pages(cfg: dict, languages: list[str] | None = None,
               count_tokens: Callable[[str], int] | None = None) -> list[dict]:
    corpus = cfg["corpus"]
    languages = languages or corpus["languages"]
    raw_dir = resolve(corpus["raw_dir"])
    clean_dir = resolve(corpus["clean_dir"])
    min_tokens = corpus.get("min_view_tokens", 50)
    snippets = load_snippets(cfg)
    pages = []
    for path in page_paths(cfg):
        raw = path.read_text(encoding="utf-8")
        slug = page_slug(path, cfg)
        for language in languages:
            text, meta = clean_page(raw, snippets, language)
            if count_tokens and count_tokens(text) < min_tokens:
                log.warning("skip %s view of %s: under %d tokens", language, slug, min_tokens)
                continue
            out = clean_dir / language / f"{slug}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            pages.append({
                "path": path.relative_to(raw_dir).as_posix(),
                "slug": slug,
                "language": language,
                "url": f"{corpus['url_base']}/{corpus['url_segment'][language]}/{slug}",
                "title": meta.get("title") or slug.rsplit("/", 1)[-1],
                "text": text,
            })
    return pages
