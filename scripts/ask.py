import argparse

from documentor.config import load_config
from documentor.pipeline import Answer, answer
from documentor.retrieval.hybrid import trace_line


def footer(a: Answer) -> str:
    t, u = a.timings, a.usage
    parts = [f"retrieve {t['retrieve_ms']:,} ms", f"generate {t['generate_ms']:,} ms", f"wait {t['wait_ms']:,} ms",
             f"total {t['total_ms']:,} ms"]
    if u:
        parts.append(f"tokens {u['prompt_tokens']:,} in / {u['completion_tokens']:,} out")
    parts += ["cache hit" if a.cache_hit else "cache miss", f"strategy {a.strategy}", f"model {a.model}"]
    return " · ".join(parts)


def show_blocks(a: Answer) -> None:
    print("Context:")
    for b in a.blocks:
        flag = "  (truncated)" if b["truncated"] else ""
        print(f"[{b['n']}] {b['score']:.3f}  {b['heading_path']} ({b['page_slug']})  {b['tokens']} tok{flag}")
        print(f"     {trace_line(a.retrieved[b['n'] - 1])}")
    print()


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Ask DocuMentor a question about LangGraph.")
    ap.add_argument("question")
    ap.add_argument("--language", choices=cfg["corpus"]["languages"], default=cfg["corpus"]["default_language"])
    ap.add_argument("--strategy", choices=["recursive", "parent_child"], default=cfg["retrieval"]["strategy"])
    ap.add_argument("--k", type=int, default=cfg["retrieval"]["top_k"])
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--show-reasoning", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="skip reading the cache (still writes)")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default=cfg["retrieval"]["mode"])
    ap.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=cfg["retrieval"]["rerank"])
    args = ap.parse_args()
    cfg["retrieval"] |= {"mode": args.mode, "rerank": args.rerank}  # search() and the query log read these

    a = answer(args.question, language=args.language, strategy=args.strategy, k=args.k,
               show_reasoning=args.show_reasoning, use_cache=not args.no_cache)

    if args.show_context and a.blocks:
        show_blocks(a)
    if args.show_reasoning and a.reasoning:
        print(f"Reasoning:\n{a.reasoning}\n")

    print(f"Answer:\n{a.text}\n")
    if a.refused:
        print(f"Refused ({a.refusal_reason}). Top retrieved:")
        for r in a.retrieved[:3]:
            print(f"  {r['score']:.3f}  {r['heading_path']} ({r['page_slug']})")
    else:
        print("Sources:")
        for s in a.sources:
            print(f"[{s['n']}] {s['heading_path']} — {s['url']}")
    if a.invalid_citations:
        print(f"\nWARNING: removed invalid citations {a.invalid_citations}")
    if a.uncited and not a.refused:
        print("\nWARNING: answer has no valid citations")
    print(f"\n{footer(a)}")


if __name__ == "__main__":
    main()
