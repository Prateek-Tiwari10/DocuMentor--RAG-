import argparse
import sys

from documentor.config import load_config
from documentor.conversation import Conversation
from documentor.pipeline import Answer, answer

HELP = "Commands: /lang python|js|auto, /reset, /history, /exit"


def info_line(a: Answer) -> str:
    t = a.timings
    parts = [f'standalone: "{a.standalone_question}"', f"language {a.language} ({a.language_reason})"]
    if a.rewrite_mode:
        rewrite = f"rewrite {a.rewrite_mode}"
        if a.rewrite_mode == "llm" and a.rewrite_ms:
            rewrite += f" {a.rewrite_ms:,} ms"
        parts.append(rewrite + (" (fell back to concat)" if a.rewrite_fallback else ""))
    parts += [f"retrieve {t['retrieve_ms']:,} ms", f"generate {t['generate_ms']:,} ms"]
    if t.get("wait_ms"):
        parts.append(f"wait {t['wait_ms']:,} ms")
    if a.cache_hit:
        parts.append("cache hit")
    return " · ".join(parts)


def show(a: Answer) -> None:
    print(a.text)
    if a.sources:
        print("\nSources:")
        for s in a.sources:
            print(f"[{s['n']}] {s['heading_path']} — {s['url']}")
    print(f"\n{info_line(a)}")


def command(line: str, conv: Conversation) -> Conversation | None:
    """Handle a /command. Returns the conversation to continue with, or None to exit."""
    name, _, arg = line.partition(" ")
    arg = arg.strip().lower()
    if name == "/exit":
        return None
    if name == "/reset":
        print("New conversation.")
        return Conversation()
    if name == "/lang":
        if arg not in ("python", "js", "auto"):
            print("Usage: /lang python|js|auto")
        else:
            conv.override = None if arg == "auto" else arg
            print(f"Language: {arg}")
        return conv
    if name == "/history":
        for i, t in enumerate(conv.turns, 1):
            print(f"{i}. [{t.language}] {t.question}\n   standalone: {t.standalone}")
        if not conv.turns:
            print("No turns yet.")
        return conv
    print(HELP)
    return conv


def messages(script: str | None):
    if script:
        with open(script, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    print(f"\n> {line.strip()}")
                    yield line.strip()
        return
    while True:
        try:
            yield input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return


def main() -> None:
    load_config()
    ap = argparse.ArgumentParser(description="Chat with DocuMentor: follow-ups are rewritten into standalone questions.")
    ap.add_argument("--script", help="run one message per line from this file, without prompting")
    ap.add_argument("--no-cache", action="store_true", help="skip reading the answer cache (still writes)")
    args = ap.parse_args()
    conv = Conversation()
    if not args.script:
        print(f"DocuMentor chat. {HELP}")
    for message in messages(args.script):
        if not message:
            continue
        if message.startswith("/"):
            conv = command(message, conv)
            if conv is None:
                return
            continue
        result = answer(message, conversation=conv, use_cache=not args.no_cache)
        conv.add(result)
        show(result)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
