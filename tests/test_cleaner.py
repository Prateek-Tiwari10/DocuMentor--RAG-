from documentor.ingest.cleaner import clean_page

PAGE = """---
title: Memory
description: How memory works
---

Shared intro text.

:::python
```python
from langgraph.checkpoint.memory import InMemorySaver
```
:::

:::js
```typescript
import { MemorySaver } from "@langchain/langgraph";
```
:::

Shared closing text.
"""

TABS = """---
title: Tabs
---

<Tabs>
  <Tab title="Postgres">
    :::python
    ```python
    from langgraph.checkpoint.postgres import PostgresSaver
    ```
    :::
    :::js
    ```typescript
    import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
    ```
    :::
  </Tab>
</Tabs>
"""

SNIPPETS_PAGE = """---
title: Snippets
---
import XyzPy from '/snippets/code-samples/xyz-py.mdx';
import XyzJs from '/snippets/code-samples/xyz-js.mdx';

Before.

<XyzPy />

<XyzJs />

After.
"""

SNIPPETS = {
    "xyz-py": "```python\nprint('python snippet')\n```\n",
    "xyz-js": "```typescript\nconsole.log('js snippet');\n```\n",
}


def test_python_view_drops_js_block():
    text, meta = clean_page(PAGE, {}, "python")
    assert meta["title"] == "Memory"
    assert "InMemorySaver" in text
    assert "MemorySaver }" not in text and "typescript" not in text
    assert ":::" not in text


def test_js_view_drops_python_block():
    text, _ = clean_page(PAGE, {}, "js")
    assert 'import { MemorySaver } from "@langchain/langgraph";' in text
    assert "InMemorySaver" not in text and "```python" not in text
    assert ":::" not in text


def test_shared_text_in_both_views():
    py, _ = clean_page(PAGE, {}, "python")
    js, _ = clean_page(PAGE, {}, "js")
    for line in ("Shared intro text.", "Shared closing text."):
        assert line in py.splitlines() and line in js.splitlines()


def test_indented_markers_inside_tabs():
    py, _ = clean_page(TABS, {}, "python")
    js, _ = clean_page(TABS, {}, "js")
    assert "from langgraph.checkpoint.postgres import PostgresSaver" in py.splitlines()
    assert "@langchain/langgraph-checkpoint-postgres" not in py
    assert 'import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";' in js.splitlines()
    assert "langgraph.checkpoint.postgres" not in js
    assert ":::" not in py and ":::" not in js


def test_tab_title_becomes_bold_and_tags_disappear():
    py, _ = clean_page(TABS, {}, "python")
    assert "**Postgres**" in py
    assert "<Tab" not in py and "</Tab" not in py


def test_empty_tab_shell_is_removed():
    page = '<Tabs>\n<Tab title="JS only">\n:::js\nonly js text\n:::\n</Tab>\n</Tabs>\n\nShared.\n'
    py, _ = clean_page(page, {}, "python")
    assert "JS only" not in py and "only js text" not in py
    js, _ = clean_page(page, {}, "js")
    assert "**JS only**" in js


def test_snippet_imports_by_language():
    py, _ = clean_page(SNIPPETS_PAGE, SNIPPETS, "python")
    js, _ = clean_page(SNIPPETS_PAGE, SNIPPETS, "js")
    assert "print('python snippet')" in py and "js snippet" not in py
    assert "console.log('js snippet');" in js and "python snippet" not in js
    for view in (py, js):
        assert "import " not in view and "<Xyz" not in view


def test_code_highlight_annotation_removed():
    page = "```python\nx = 1  # [!code highlight]\n```\n\n```ts\nconst y = 2; // [!code ++]\n```\n"
    py, _ = clean_page(page, {}, "python")
    js, _ = clean_page(page, {}, "js")
    assert "x = 1\n" in py and "[!code" not in py
    assert "const y = 2;\n" in js and "[!code" not in js


def test_code_fence_contents_untouched():
    code = '```python\nif x < y and y > 0:\n    s = ":::"\n    tag = "<Tab title=\\"a\\">"\nimport os\n```'
    page = f"Intro.\n\n{code}\n"
    py, _ = clean_page(page, {}, "python")
    assert code in py


def test_relative_links_become_absolute_per_language():
    page = "See [persistence](/oss/langgraph/persistence) and [traces](/langsmith/observability).\n"
    py, _ = clean_page(page, {}, "python")
    js, _ = clean_page(page, {}, "js")
    assert "(https://docs.langchain.com/oss/python/langgraph/persistence)" in py
    assert "(https://docs.langchain.com/oss/javascript/langgraph/persistence)" in js
    assert "(https://docs.langchain.com/langsmith/observability)" in py
