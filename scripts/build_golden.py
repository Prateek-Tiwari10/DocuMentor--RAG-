import json
import sys

from documentor.config import load_config, resolve
from documentor.evaluation import expected_entries, first_hit_rank
from documentor.evaluation.golden import auto_check, expected_text, load_golden, read_page, section_exists
from documentor.evaluation.ragas_eval import BudgetExhausted, budget_message, budget_sleep, is_daily_limit, reset_hint
from documentor.generation.llm import GroqLLM, get_encoding
from documentor.generation.prompt import truncate
from documentor.retrieval.dense import search

PROMPT = ("Using only the documentation section below, write a short, correct answer to the question. "
          "Use only names that appear in the section. "
          "Write 2 to 5 sentences and at most one short code snippet.")
STRATEGIES = ["recursive", "parent_child"]
V120_DELTA = 'langgraph v1.2.0, May 12, 2026: "DeltaChannel (beta)"'
V120_EVENTS = 'langgraph v1.2.0, May 12, 2026: "New event streaming API (beta)", version="v3" for stream_events()'


def new_question(qid: str, question: str, slug: str, section: str, evidence: str | None = None) -> dict:
    item = {"id": qid, "question": question, "expected_slug": slug, "expected_section": section,
            "language": "python", "verified": False, "post_cutoff": True if evidence else None}
    return item | ({"cutoff_evidence": evidence} if evidence else {})


NEW_QUESTIONS = [
    new_question("q16", "Why does invoking my graph raise InvalidUpdateError saying it expected a dict but got a list?",
                 "langgraph/errors/INVALID_GRAPH_NODE_RETURN_VALUE", ""),
    new_question("q17", "LangGraph tells me I'm using persistence without a checkpointer. Where am I supposed to pass one in?",
                 "langgraph/errors/MISSING_CHECKPOINTER", "Troubleshooting"),
    new_question("q18", "My agent stopped mid tool call and now invoking it fails because an AIMessage has tool_calls "
                        "with no matching ToolMessage. How do I get that thread working again?",
                 "langgraph/errors/INVALID_CHAT_HISTORY", "Troubleshooting"),
    new_question("q19", "Using stream_events, how can I get the complete text and the token usage of each model reply "
                        "rather than individual tokens?",
                 "langgraph/event-streaming", "Stream messages", V120_EVENTS),
    new_question("q20", "I wrote my own checkpoint saver. What does it have to provide so channels that only store "
                        "deltas can rebuild their values when a checkpoint loads?",
                 "langgraph/checkpointers", "Delta channel support", V120_DELTA),
    new_question("q21", "My long-running graph spends too much time persisting state. How do I trade some crash safety "
                        "for speed, and what are the options?",
                 "langgraph/checkpointers", "Durability modes"),
    new_question("q22", "How can I unit test one node of a compiled graph without running the whole thing?",
                 "langgraph/test", "Testing individual nodes and edges"),
    new_question("q23", "In a functional API workflow, how do I return one value to the caller but store a different "
                        "value for the next invocation?",
                 "langgraph/functional-api", "entrypoint.final"),
    new_question("q24", "How do I stop values like social security numbers from showing up in the traces my graph "
                        "sends to LangSmith?",
                 "langgraph/observability", "anonymizers"),
    new_question("q25", "How do I start a variable number of worker nodes at runtime, each with its own input, and "
                        "collect all their outputs in one place?",
                 "langgraph/workflows-agents", "Creating workers in LangGraph"),
]
SPARES = [
    ("I get the MULTIPLE_SUBGRAPHS error when my node calls the same subgraph several times with "
     "checkpointer=True. What are my choices?", "langgraph/errors/MULTIPLE_SUBGRAPHS", "Troubleshooting"),
    ("What goes in the langgraph.json file when I set up an app for deployment?",
     "langgraph/application-structure", "Configuration file"),
    ("How do I pass read-only settings such as a user ID into my graph when I invoke it?",
     "concepts/context", "Static runtime context"),
    ("What does ToolNode do when I use it in an agent graph?", "langgraph/workflows-agents", "ToolNode"),
]


def ranks_for(item: dict, cfg: dict) -> dict:
    return {s: first_hit_rank(search(item["question"], s, k=cfg["retrieval"]["top_k"],
                                     language=item.get("language")), item) for s in STRATEGIES}


def write_reference(item: dict, llm, read, cfg: dict, fresh: bool) -> tuple[str, list[str]]:
    section = expected_text(item, read)
    enc = get_encoding(cfg["generation"]["budget_encoding"])
    limit = cfg["generation"]["context_max_tokens"]
    shown = truncate(section, limit, enc) if len(enc.encode(section)) > limit else section
    content = f"{PROMPT}\n\nDocumentation section:\n{shown}\n\nQuestion: {item['question']}"
    out = llm.complete([{"role": "user", "content": content}], use_cache=not fresh)
    problems = auto_check(item, out.text, read)
    if out.finish_reason != "stop":
        problems.append(f"finish_reason={out.finish_reason}")
    return out.text.strip(), problems


def save(items: list[dict], path) -> None:
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")


def add_new_questions(items: list[dict], cfg: dict) -> None:
    known = {i["id"] for i in items}
    for item in NEW_QUESTIONS:
        if item["id"] not in known:
            items.append(item | {"draft_ranks": ranks_for(item, cfg)})


def next_spare(old: dict, used: set[str], cfg: dict) -> dict | None:
    for question, slug, section in SPARES:
        if question not in used:
            used.add(question)
            item = new_question(old["id"], question, slug, section)
            return item | {"draft_ranks": ranks_for(item, cfg)}
    return None


def main() -> None:
    cfg = load_config()
    ev = cfg["evaluation"]
    path = resolve(ev["golden_path"])
    read = lambda language, slug: read_page(resolve(cfg["corpus"]["clean_dir"]), language, slug)
    items = load_golden(path)
    add_new_questions(items, cfg)
    save(items, path)
    llm = GroqLLM(cfg, sleep=budget_sleep(ev["budget_max_wait_s"]))
    new_ids = {i["id"] for i in NEW_QUESTIONS}
    used: set[str] = {i["question"] for i in items}
    dropped = []

    for pos, item in enumerate(items):
        while not item.get("auto_checked"):
            if not section_exists(item, read):
                problems = ["expected section not found in the cleaned page"]
            else:
                reference, problems = write_reference(item, llm, read, cfg, fresh=False)
                if problems:
                    print(f"  {item['id']}: regenerating once ({'; '.join(problems)})")
                    reference, problems = write_reference(item, llm, read, cfg, fresh=True)
            if not problems:
                item.update(reference=reference, auto_checked=True)
                print(f"  {item['id']}: reference ok")
            elif item["id"] in new_ids:
                dropped.append({"id": item["id"], "question": item["question"], "problems": problems})
                print(f"  {item['id']}: dropped ({'; '.join(problems)})")
                item = next_spare(item, used, cfg)
                if item is None:
                    sys.exit("No spare questions left; add more to SPARES.")
                items[pos] = item
            else:
                item.update(reference=reference, auto_checked=False, auto_check_problems=problems)
                print(f"  {item['id']}: FAILED checks ({'; '.join(problems)})")
                break
            save(items, path)

    print(f"\n{'id':<5} {'page':<46} {'section':<34} {'post_cutoff':<11} {'checked':<7} {'rec':>4} {'p_c':>4}")
    for item in items:
        entry = expected_entries(item)[0]
        ranks = item.get("draft_ranks") or ranks_for(item, cfg)
        cutoff = "-" if item.get("post_cutoff") is None else str(item["post_cutoff"]).lower()
        print(f"{item['id']:<5} {entry['slug'][:45]:<46} {(entry.get('section') or '(page)')[:33]:<34} {cutoff:<11} "
              f"{str(bool(item.get('auto_checked'))):<7} {ranks['recursive'] or '-':>4} {ranks['parent_child'] or '-':>4}")
    failed = [i["id"] for i in items if not i.get("auto_checked")]
    print(f"\n{len(items)} questions, {sum(not i.get('verified') for i in items)} with verified: false, "
          f"failed checks: {failed or 'none'}, dropped and replaced: {dropped or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        main()
    except BudgetExhausted as e:
        print(budget_message(e))
        sys.exit(3)
    except Exception as e:
        if not is_daily_limit(str(e)):
            raise
        print(budget_message(BudgetExhausted(reset_hint(str(e)))))
        sys.exit(3)
