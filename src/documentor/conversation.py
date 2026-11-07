import copy
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from documentor.config import load_config
from documentor.generation.llm import LLM, GenerationError, GroqLLM, get_encoding

if TYPE_CHECKING:
    from documentor.pipeline import Answer

MODES = ("none", "concat", "llm")
QUOTES = "\"'`“”‘’"
REWRITE_PROMPT = """Rewrite the user's last message as one standalone question about LangGraph that can be understood without the conversation.
- Keep code names exactly as written.
- If the last message is already standalone or changes the topic, return it unchanged.
- Reply with the question only.

Conversation:
{history}

Last message: {message}"""


@dataclass
class Turn:
    question: str
    standalone: str
    language: str
    answer: str
    sources: list[dict]


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turns: list[Turn] = field(default_factory=list)
    language: str | None = None
    override: str | None = None    # set by /lang in chat, or by the Phase 6 UI toggle

    def add(self, answer: "Answer") -> None:
        self.turns.append(Turn(answer.question, answer.standalone_question or answer.question, answer.language,
                               answer.text, answer.sources))
        self.language = answer.language

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(id=data["id"], turns=[Turn(**t) for t in data.get("turns", [])], language=data.get("language"),
                   override=data.get("override"))


def rewrite_cfg(cfg: dict) -> dict:
    """The answer model's LLM settings with the rewrite model swapped in, so it gets its own limiter."""
    conv = cfg["conversation"]
    out = copy.deepcopy(cfg)
    out["llm"] |= {"model": conv["rewrite_model"], "temperature": conv["rewrite_temperature"],
                   "reasoning_effort": conv["rewrite_reasoning_effort"],
                   "max_completion_tokens": conv["rewrite_max_tokens"], "tpm_limit": conv["rewrite_tpm_limit"]}
    return out


_rewrite_llm: LLM | None = None


def default_rewrite_llm(cfg: dict) -> LLM:
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = GroqLLM(rewrite_cfg(cfg))
    return _rewrite_llm


def cut_tokens(text: str, n: int, encoding: str) -> str:
    enc = get_encoding(encoding)
    ids = enc.encode(text)
    return text if len(ids) <= n else enc.decode(ids[:n])


def history_text(conversation: Conversation, cfg: dict) -> str:
    conv = cfg["conversation"]
    encoding = cfg["generation"]["budget_encoding"]
    lines = []
    for turn in conversation.turns[-conv["history_turns"]:]:
        lines += [f"User: {turn.question}", f"Assistant: {cut_tokens(turn.answer, conv['history_answer_tokens'], encoding)}"]
    return "\n".join(lines)


def rewrite_messages(message: str, conversation: Conversation, cfg: dict) -> list[dict]:
    return [{"role": "user", "content": REWRITE_PROMPT.format(history=history_text(conversation, cfg), message=message)}]


def make_standalone(message: str, conversation: Conversation | None, mode: str, llm: LLM | None = None,
                    cfg: dict | None = None) -> tuple[str, dict]:
    """A question the retriever can match on its own. The first turn is returned unchanged in every mode."""
    if mode not in MODES:
        raise ValueError(f"unknown rewrite mode: {mode}")
    cfg = cfg or load_config()
    info = {"mode": mode, "rewrite_ms": 0, "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "fallback": False, "cache_hit": False}
    if not conversation or not conversation.turns or mode == "none":
        return message, info
    concat = f"{conversation.turns[-1].question} {message}"
    if mode == "concat":
        return concat, info

    start = time.perf_counter()
    llm = llm or default_rewrite_llm(cfg)
    try:
        out = llm.complete(rewrite_messages(message, conversation, cfg))
    except GenerationError:  # the whole budget went to reasoning
        info["rewrite_ms"] = int((time.perf_counter() - start) * 1000)
        return concat, info | {"fallback": True}
    info |= {"rewrite_ms": max(0, int((time.perf_counter() - start) * 1000) - out.wait_ms), "latency_ms": out.latency_ms,
             "prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens, "cache_hit": out.cache_hit}
    text = out.text.strip().strip(QUOTES).strip()
    if not text or out.finish_reason == "length":
        return concat, info | {"fallback": True}
    return text, info
