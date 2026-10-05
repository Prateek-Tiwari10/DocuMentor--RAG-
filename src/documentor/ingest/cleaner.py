import re
import textwrap

import yaml

HOST = "https://docs.langchain.com"
URL_SEGMENT = {"python": "python", "js": "javascript"}
LANG_SUFFIX = {"python": "-py", "js": "-js"}
LANG_MARKERS = {":::python": "python", ":::js": "js"}
FENCE_LANGS = {"python": {"python", "py", "pycon"}, "js": {"typescript", "ts", "tsx", "javascript", "js", "jsx"}}

# Blocks whose whole content is dropped (an LLM prompt for coding agents, not docs).
DROP_BLOCKS = {"Prompt"}
HTML_TAGS = "div|span|a|ul|ol|li|table|thead|tbody|tr|td|th|img|iframe|br|p|code|strong|em|b|i|sup|sub|video|source"
ATTRS = r"""(?:"[^"]*"|'[^']*'|\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|[^<>"'{}])*"""
TAG_RE = re.compile(rf"<(/?)([A-Z][\w.]*|(?:{HTML_TAGS})\b)({ATTRS}?)(/?)>", re.S)
TITLE_RE = re.compile(r"""\btitle=(?:"([^"]*)"|'([^']*)')""")
IMPORT_RE = re.compile(r"""^\s*import\s+(\w+)\s+from\s+['"]([^'"]+)['"];?\s*$""")
USAGE_RE = re.compile(r"^(\s*)<(\w+)\s*/>\s*$")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
CODE_ANNOT_RE = re.compile(r"\s*(?:#|//|<!--|/\*)\s*\[!code[^\]]*\]\s*(?:-->|\*/)?")
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.S)
API_REF_RE = re.compile(r"@\[([^\]\n]*)\](?:\[[^\]\n]*\])?")
LINK_RE = re.compile(r"(\]\(|href=\")(/(?:oss|langsmith|langgraph-platform|images)[^)\"\s]*)")
TITLE_MARK = "\x00TITLE:"
END_MARK = "\x00END"


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = yaml.safe_load(text[3:end]) or {}
    return meta, text[end + 4:]


def fence_spans(lines: list[str]) -> list[bool]:
    """Return a flag per line: True if the line is part of a fenced code block (fences included)."""
    flags, open_fence = [], None
    for line in lines:
        m = FENCE_RE.match(line)
        if open_fence is None:
            if m:
                open_fence = m.group(2)
            flags.append(open_fence is not None)
        else:
            flags.append(True)
            if m and m.group(2).startswith(open_fence) and not m.group(3).strip():
                open_fence = None
    return flags


def snippet_language(path: str) -> str | None:
    stem = path.rsplit("/", 1)[-1].removesuffix(".mdx")
    for lang, suffix in LANG_SUFFIX.items():
        if stem.endswith(suffix):
            return lang
    return None


def inline_snippets(text: str, snippets: dict[str, str], language: str) -> str:
    lines = text.splitlines()
    flags = fence_spans(lines)
    imports = {}
    for line, in_fence in zip(lines, flags):
        m = IMPORT_RE.match(line)
        if m and not in_fence:
            imports[m.group(1)] = m.group(2)
    out = []
    for line, in_fence in zip(lines, flags):
        m = USAGE_RE.match(line)
        if in_fence or not m or m.group(2) not in imports:
            out.append(line)
            continue
        path = imports[m.group(2)]
        stem = path.rsplit("/", 1)[-1].removesuffix(".mdx")
        lang = snippet_language(path)
        if "/code-samples/" in path and stem in snippets and lang in (None, language):
            out.append(textwrap.indent(snippets[stem].strip("\n"), m.group(1)))
        else:
            out.append("")
    return "\n".join(out)


def fence_language(line: str) -> str | None:
    m = FENCE_RE.match(line)
    info = m.group(3).split()[0].lower() if m and m.group(3).split() else ""
    return next((lang for lang, names in FENCE_LANGS.items() if info in names), None)


def filter_lines(text: str, language: str) -> str:
    """Language blocks, import/export lines and code annotations, one line at a time."""
    lines = text.splitlines()
    flags = fence_spans(lines)
    out, block, skip_fence = [], None, False
    for i, (line, in_fence) in enumerate(zip(lines, flags)):
        s = line.strip()
        # A shared-region fence tagged with the other language belongs to that view only.
        if in_fence and (i == 0 or not flags[i - 1]) and block is None:
            lang = fence_language(line)
            skip_fence = lang is not None and lang != language
        if not in_fence:
            skip_fence = False
        if skip_fence:
            continue
        if not in_fence:
            if s in LANG_MARKERS and block is None:
                block = LANG_MARKERS[s]
                continue
            if s == ":::":
                block = None
                continue
        if block is not None and block != language:
            continue
        if in_fence:
            out.append(CODE_ANNOT_RE.sub("", line) if "[!code" in line else line)
        elif not re.match(r"^\s*(import|export)\s", line):
            out.append(line)
    return "\n".join(out)


def mask_inline_code(text: str) -> tuple[str, list[str]]:
    saved = []

    def keep(m):
        saved.append(m.group(0))
        return f"\x01{len(saved) - 1}\x01"

    return re.sub(r"`[^`\n]+`", keep, text), saved


def unmask(text: str, saved: list[str]) -> str:
    return re.sub(r"\x01(\d+)\x01", lambda m: saved[int(m.group(1))], text)


def strip_tags(text: str) -> str:
    text = MDX_COMMENT_RE.sub("", text)
    for name in DROP_BLOCKS:
        text = re.sub(rf"<{name}\b{ATTRS}?>.*?</{name}>", "", text, flags=re.S)
        text = re.sub(rf"<{name}\b{ATTRS}?/>", "", text, flags=re.S)
    text, saved = mask_inline_code(text)

    def replace(m):
        closing, name, attrs, self_closing = m.groups()
        if closing and name[0].isupper():
            return f"\n{END_MARK}\n"
        if closing or self_closing:
            return "\n" if name in ("li", "tr", "br") else " " if name in ("td", "th") else ""
        t = TITLE_RE.search(attrs)
        if t and name[0].isupper():
            return f"\n{TITLE_MARK}{t.group(1) or t.group(2)}\n"
        return "\n- " if name == "li" else ""

    text = TAG_RE.sub(replace, text)
    return unmask(text, saved)


def drop_empty_titles(lines: list[str]) -> list[str]:
    """A tab title whose tab lost all content (other-language block removed) is dropped."""
    out = []
    for i, line in enumerate(lines):
        if line.strip() == END_MARK:
            continue
        if line.strip().startswith(TITLE_MARK):
            nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            if not nxt or nxt.startswith(TITLE_MARK) or nxt == END_MARK:
                continue
            out.append(f"**{line.strip()[len(TITLE_MARK):]}**")
        else:
            out.append(line)
    return out


def dedent_blocks(text: str) -> str:
    """Dedent each fenced block by its fence indent, and each prose paragraph by its common indent."""
    lines = text.splitlines()
    flags = fence_spans(lines)
    out, para, fence_indent = [], [], None

    def flush():
        if para:
            out.extend(textwrap.dedent("\n".join(para)).splitlines())
            para.clear()

    for line, in_fence in zip(lines, flags):
        if in_fence:
            flush()
            m = FENCE_RE.match(line)
            if fence_indent is None:
                fence_indent = len(m.group(1))
                out.append(line[fence_indent:])
                continue
            lead = len(line) - len(line.lstrip(" "))
            out.append(line[min(lead, fence_indent):])
            if m and not m.group(3).strip():
                fence_indent = None
        elif not line.strip():
            flush()
            out.append("")
        else:
            para.append(line)
    flush()
    return "\n".join(out)


def apply_outside_fences(text: str, fn) -> str:
    """Run fn on each run of non-fence lines, leaving code fences untouched."""
    lines = text.splitlines()
    flags = fence_spans(lines)
    out, run = [], []
    for line, in_fence in zip(lines, flags):
        if in_fence:
            if run:
                out.append(fn("\n".join(run)))
                run = []
            out.append(line)
        else:
            run.append(line)
    if run:
        out.append(fn("\n".join(run)))
    return "\n".join(out)


def absolutize_links(text: str, language: str) -> str:
    segment = URL_SEGMENT[language]

    def fix(m):
        prefix, path = m.groups()
        if path.startswith("/oss/") and not path.startswith(("/oss/images", f"/oss/{segment}/")):
            path = f"/oss/{segment}/" + path[len("/oss/"):]
        return prefix + HOST + path

    return LINK_RE.sub(fix, text)


def clean_page(text: str, snippets: dict[str, str], language: str) -> tuple[str, dict]:
    meta, body = split_frontmatter(text)
    body = inline_snippets(body, snippets, language)
    body = filter_lines(body, language)
    body = apply_outside_fences(body, lambda t: absolutize_links(API_REF_RE.sub(r"\1", strip_tags(t)), language))
    lines = drop_empty_titles([l.rstrip() for l in body.splitlines()])
    body = dedent_blocks("\n".join(lines))
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
    meta.setdefault("title", "")
    return body, meta
