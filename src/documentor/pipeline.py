import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime

from documentor.config import load_config, resolve
from documentor.conversation import Conversation, make_standalone
from documentor.generation.citations import parse_answer, sources_for
from documentor.generation.llm import LLM, get_llm
from documentor.generation.prompt import build_context, build_messages
from documentor.language import detect_language
from documentor.retrieval import TRACE_FIELDS


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
    standalone_question: str | None = None   # what retrieval and generation saw
    language_reason: str | None = None
    rewrite_mode: str | None = None
    rewrite_ms: int = 0
    rewrite_fallback: bool = False


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
           use_cache: bool = True, cfg: dict | None = None, conversation: Conversation | None = None,
           rewrite_llm: LLM | None = None) -> Answer:
    """With a conversation: route the language and rewrite a follow-up into a standalone question first.
    Without one, this is the Phase 4 single-question path. History never reaches the answer model."""
    cfg = cfg or load_config()
    gen = cfg["generation"]
    standalone, reason, info = question, "override" if language else "default", None
    if conversation is not None:
        language, reason = detect_language(question, previous=conversation.language,
                                           override=language or conversation.override, cfg=cfg)
        standalone, info = make_standalone(question, conversation, cfg["conversation"]["rewrite"], llm=rewrite_llm, cfg=cfg)
    language = language or cfg["corpus"]["default_language"]
    strategy = strategy or cfg["retrieval"]["strategy"]
    k = k or cfg["retrieval"]["top_k"]
    if search_fn is None:
        from documentor.retrieval.dense import search as search_fn

    start = time.perf_counter()
    results = search_fn(standalone, strategy, k=k, language=language)
    retrieve_ms = ms(start)
    retrieved = [{"id": r.get("id") or r.get("parent_id"), "score": r.get("score"),
                  "heading_path": r.get("heading_path"), "page_slug": r.get("page_slug"),
                  **{f: r.get(f) for f in TRACE_FIELDS}} for r in results]

    base = dict(question=question, language=language, strategy=strategy, invalid_citations=[],
                uncited=False, reasoning=None, retrieved=retrieved, standalone_question=standalone,
                language_reason=reason, rewrite_mode=info["mode"] if info else None,
                rewrite_ms=info["rewrite_ms"] if info else 0, rewrite_fallback=bool(info and info["fallback"]))
    min_score = gen.get("min_score")
    if min_score is not None and (not results or results[0]["score"] < min_score):
        result = Answer(**base, text=gen["refusal_message"], refused=True, refusal_reason="low_score",
                        sources=[], blocks=[], timings={"retrieve_ms": retrieve_ms, "wait_ms": 0, "generate_ms": 0,
                                                        "total_ms": ms(start)},
                        usage={}, cache_hit=False, model=cfg["llm"]["model"])
        log_answer(result, cfg)
        return result

    context, blocks = build_context(results, gen["context_max_tokens"], gen["budget_encoding"])
    messages = build_messages(standalone, context, language)
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
    retrieval = cfg.get("retrieval", {})
    row["config"] = {
        "model": cfg["llm"]["model"],
        "temperature": cfg["llm"]["temperature"],
        "reasoning_effort": cfg["llm"]["reasoning_effort"],
        "context_max_tokens": cfg["generation"]["context_max_tokens"],
        "retrieval_mode": retrieval.get("mode"),
        "rerank": retrieval.get("rerank"),
        "reranker_model": retrieval.get("reranker_model") if retrieval.get("rerank") else None,
    }
    path = resolve(cfg["logging"]["query_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
