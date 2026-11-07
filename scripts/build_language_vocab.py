from collections import Counter

from documentor.config import load_config, resolve
from documentor.evaluation import load_jsonl
from documentor.language import build_language_vocab, detect_language


def top(entries: dict, n: int = 20) -> str:
    return ", ".join(f"{w} ({c})" for w, c in list(entries.items())[:n])


def main() -> None:
    cfg = load_config()
    vocab = build_language_vocab(cfg)
    print(f"python_only: {len(vocab['python_only']):,} identifiers\n  {top(vocab['python_only'])}")
    print(f"js_only: {len(vocab['js_only']):,} identifiers\n  {top(vocab['js_only'])}")
    print(f"Saved {cfg['language']['vocab_path']} (corpus {vocab['corpus_commit'][:7]})")

    golden = load_jsonl(resolve(cfg["evaluation"]["golden_path"]))
    routes = {q["id"]: detect_language(q["question"], vocab=vocab, cfg=cfg) for q in golden}
    counts = Counter(lang for lang, _ in routes.values())
    print(f"\nGolden check: {dict(counts)} of {len(golden)} questions")
    for qid, (lang, reason) in routes.items():
        if lang != "python" or reason != "default":
            print(f"  {qid}: {lang} ({reason})")
    wrong = [qid for qid, (lang, _) in routes.items() if lang != "python"]
    if wrong:
        raise SystemExit(f"Golden questions routed away from Python: {', '.join(wrong)}. Fix the rule, not the golden set.")


if __name__ == "__main__":
    main()
