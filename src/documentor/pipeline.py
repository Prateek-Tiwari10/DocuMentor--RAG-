import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime

from documentor.config import load_config, resolve
from documentor.generation.citations import parse_answer, sources_for
from documentor.generation.llm import LLM, get_llm
from documentor.generation.prompt import build_context, build_messages


@dataclass
class Answer:
    question: str
    language: str
    strategy: str
    text: str                 # final answer, or refusal_message
    refused: bool
    refusal_reason: str | None   # "model" | "low_score" | None
    sources: list[dict]
    blocks: list[dict]        # everything that was in the context
    invalid_citations: list[int]
    uncited: bool
    timings: dict             # retrieve_ms, wait_ms, generate_ms (excludes wait_ms), total_ms (includes it)
    usage: dict               # prompt_tokens, completion_tokens, reasoning_tokens?, remaining_tokens?
    cache_hit: bool
    model: str
    reasoning: str | None
    retrieved: list[dict]     # top-k ids and scores, before the context budget


_llm: LLM | None = None


def default_llm(cfg: dict) -> LLM:
    global _llm
    if _llm is None:
        _llm = get_llm(cfg)
    return _llm


def ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def answer(question: str, language: str | None = None, strategy: str | None = None, k: int | None = None,
           llm: LLM | None = None, search_fn=None, show_reasoning: bool = False,
           use_cache: bool = True, cfg: dict | None = None) -> Answer:
    cfg = cfg or load_config()
    gen = cfg["generation"]
    language = language or cfg["corpus"]["default_language"]
    strategy = strategy or cfg["retrieval"]["strategy"]
    k = k or cfg["retrieval"]["top_k"]
    if search_fn is None:
        from documentor.retrieval.dense import search as search_fn

    start = time.perf_counter()
    results = search_fn(question, strategy, k=k, language=language)
    retrieve_ms = ms(start)
    retrieved = [{"id": r.get("id") or r.get("parent_id"), "score": r.get("score"),
                  "heading_path": r.get("heading_path"), "page_slug": r.get("page_slug")} for r in results]

    base = dict(question=question, language=language, strategy=strategy, invalid_citations=[],
                uncited=False, reasoning=None, retrieved=retrieved)
    min_score = gen.get("min_score")
    if min_score is not None and (not results or results[0]["score"] < min_score):
        result = Answer(**base, text=gen["refusal_message"], refused=True, refusal_reason="low_score",
                        sources=[], blocks=[], timings={"retrieve_ms": retrieve_ms, "wait_ms": 0, "generate_ms": 0,
                                                        "total_ms": ms(start)},
                        usage={}, cache_hit=False, model=cfg["llm"]["model"])
        log_answer(result, cfg)
        return result

    context, blocks = build_context(results, gen["context_max_tokens"], gen["budget_encoding"])
    messages = build_messages(question, context, language)
    llm = llm or default_llm(cfg)
    gen_start = time.perf_counter()
    out = llm.complete(messages, show_reasoning=show_reasoning, use_cache=use_cache)
    generate_ms = max(0, ms(gen_start) - out.wait_ms)

    parsed = parse_answer(out.text, blocks, gen["refusal_sentinel"])
    usage = {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens}
    if out.reasoning_tokens is not None:
        usage["reasoning_tokens"] = out.reasoning_tokens
    if out.remaining_tokens is not None:
        usage["remaining_tokens"] = out.remaining_tokens

    base |= dict(reasoning=out.reasoning, invalid_citations=parsed.invalid, uncited=parsed.uncited)
    result = Answer(
        **base,
        text=gen["refusal_message"] if parsed.refused else parsed.text,
        refused=parsed.refused,
        refusal_reason="model" if parsed.refused else None,
        sources=[] if parsed.refused else sources_for(parsed, blocks),
        blocks=blocks,
        timings={"retrieve_ms": retrieve_ms, "wait_ms": out.wait_ms, "generate_ms": generate_ms,
                 "total_ms": ms(start)},
        usage=usage,
        cache_hit=out.cache_hit,
        model=out.model,
    )
    log_answer(result, cfg)
    return result


def log_answer(result: Answer, cfg: dict) -> None:
    row = asdict(result)
    row.pop("reasoning")
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["config"] = {
        "model": cfg["llm"]["model"],
        "temperature": cfg["llm"]["temperature"],
        "reasoning_effort": cfg["llm"]["reasoning_effort"],
        "context_max_tokens": cfg["generation"]["context_max_tokens"],
    }
    path = resolve(cfg["logging"]["query_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
