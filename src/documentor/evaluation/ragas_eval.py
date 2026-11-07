import asyncio
import hashlib
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

RAG_METRICS = ["answer_correctness", "faithfulness", "answer_relevancy", "context_precision"]
CLOSED_METRICS = ["answer_correctness", "answer_relevancy"]
CONTEXT_METRICS = {"faithfulness", "context_precision"}
REFERENCE_METRICS = {"answer_correctness", "context_precision"}
METRIC_CLASSES = {"faithfulness": "Faithfulness", "answer_correctness": "AnswerCorrectness",
                  "answer_relevancy": "AnswerRelevancy", "context_precision": "ContextPrecisionWithReference"}
TRANSIENT = re.compile(r"rate.?limit|\b429\b|timed? ?out|connection|\b50\d\b", re.I)


class BudgetExhausted(RuntimeError):
    def __init__(self, reset: str | None = None):
        super().__init__(f"daily Groq budget used up (reset: {reset})")
        self.reset = reset
        self.at = reset_time(reset)  # Groq's wait counts from now, so fix the clock time while it's still accurate


def parse_metrics(value: str) -> list[str]:
    """Comma-separated metric keys, in the standard order. Raises ValueError on an unknown key or an empty list."""
    chosen = [m.strip() for m in value.split(",") if m.strip()]
    unknown = sorted(set(chosen) - set(RAG_METRICS))
    if unknown or not chosen:
        raise ValueError(f"--metrics takes keys from {', '.join(RAG_METRICS)}; got {value!r}")
    return [m for m in RAG_METRICS if m in chosen]


def is_daily_limit(message: str) -> bool:
    return bool(re.search(r"per.day|\bTPD\b|\bRPD\b", message, re.I))


def reset_hint(message: str) -> str | None:
    m = re.search(r"try again in\s+((?:[0-9.]+[hms])+)", message)
    return m.group(1) if m else None


def reset_time(hint: str | None, now: datetime | None = None) -> str:
    """Clock time for a Groq duration like 7m12.5s; a plain description if there is none."""
    parts = re.findall(r"([0-9.]+)([hms])", hint or "")
    if not parts:
        return "the daily limit resets (Groq gave no time)"
    scale = {"h": 3600, "m": 60, "s": 1}
    delta = timedelta(seconds=sum(float(v) * scale[u] for v, u in parts) + 60)
    return f"{(now or datetime.now()) + delta:%Y-%m-%d %H:%M}"


def budget_message(err: BudgetExhausted) -> str:
    return f"Daily Groq budget used up. Rerun the same command after {err.at}; finished work is cached."


def budget_sleep(max_wait: float):
    """A sleep for GroqLLM that treats a very long retry wait as a used-up daily budget."""
    import time

    def sleep(delay: float) -> None:
        if delay > max_wait:
            raise BudgetExhausted(f"{delay:.0f}s")
        time.sleep(delay)
    return sleep


@dataclass
class Sample:
    qid: str
    question: str
    answer: str
    contexts: list[str]
    reference: str


class Scorer(Protocol):
    judge_model: str
    version: str

    def score(self, sample: Sample, metrics: list[str]) -> dict[str, float | None]: ...


def rag_samples(answers: dict[str, dict], golden: list[dict]) -> list[Sample]:
    """Answered in-scope questions only; refusals are scored by refusal correctness."""
    return [Sample(q["id"], q["question"], answers[q["id"]]["text"],
                   [b["text"] for b in answers[q["id"]]["blocks"]], q["reference"])
            for q in golden if not answers[q["id"]]["refused"]]


def closed_samples(answers: dict[str, dict], golden: list[dict]) -> list[Sample]:
    return [Sample(q["id"], q["question"], answers[q["id"]]["text"], [], q["reference"]) for q in golden]


def metrics_for(sample: Sample, metrics: list[str]) -> list[str]:
    return [m for m in metrics if sample.contexts or m not in CONTEXT_METRICS]


def cache_key(sample: Sample, metric: str, judge_model: str, version: str) -> str:
    # A metric's key leaves out inputs it never reads, so calibration scores are reused by eval_all.
    payload = [sample.qid, sample.answer, sample.contexts if metric in CONTEXT_METRICS else [],
               sample.reference if metric in REFERENCE_METRICS else "", metric, judge_model, version]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def cache_read(root, key: str) -> dict | None:
    path = Path(root) / key[:2] / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cache_write(root, key: str, entry: dict) -> None:
    path = Path(root) / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry))


def score_sample(scorer: Scorer, sample: Sample, metrics: list[str], cache_root) -> dict[str, dict]:
    """Score one metric at a time; each finished score is cached before the next call starts."""
    out = {}
    for metric in metrics_for(sample, metrics):
        key = cache_key(sample, metric, scorer.judge_model, scorer.version)
        entry = cache_read(cache_root, key)
        if entry is not None:
            out[metric] = entry | {"cached": True}
            continue
        before = getattr(scorer, "tokens", 0)
        transient = False
        try:
            value, error = scorer.score(sample, [metric])[metric], None
            if value is not None and math.isnan(value):
                value, error = None, "metric returned NaN"
        except BudgetExhausted:
            raise
        except Exception as e:
            message = str(e)
            if is_daily_limit(message):
                raise BudgetExhausted(reset_hint(message)) from e
            value, error = None, f"{type(e).__name__}: {message[:300]}"
            transient = bool(TRANSIENT.search(message))
        entry = {"value": value, "error": error, "tokens": getattr(scorer, "tokens", 0) - before,
                 "created_at": datetime.now().isoformat(timespec="seconds")}
        if not transient:
            cache_write(cache_root, key, entry)
        out[metric] = entry | {"cached": False}
    return out


def mean_score(rows: list[dict], metric: str) -> float | None:
    values = [r[metric]["value"] for r in rows if metric in r and r[metric]["value"] is not None]
    return statistics.mean(values) if values else None


def failed_metrics(rows_by_q: dict[str, dict]) -> list[dict]:
    return [{"qid": q, "metric": m, "error": e["error"]} for q, row in rows_by_q.items() for m, e in row.items()
            if e["value"] is None]


class RagasScorer:
    def __init__(self, cfg: dict, judge_model: str, embedder=None):
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        import httpx
        import ragas
        from dotenv import load_dotenv
        from openai import AsyncOpenAI
        from ragas.embeddings.base import BaseRagasEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics import collections

        from documentor.config import resolve
        from documentor.generation.llm import TokenRateLimiter, count_message_tokens
        from documentor.retrieval.dense import get_embedder

        ev = cfg["evaluation"]
        load_dotenv(resolve(".env"))
        self.judge_model = judge_model
        self.version = ragas.__version__
        self.tokens = 0
        self.budget_hit = False
        self.budget_reset: str | None = None
        self.min_requests = ev["judge_min_requests_left"]
        limiter = TokenRateLimiter(ev["judge_tpm_limit"])
        encoding = cfg["generation"]["budget_encoding"]
        self.loop = asyncio.new_event_loop()

        async def on_request(request):
            body = json.loads(request.content)
            limiter.wait(count_message_tokens(body["messages"], encoding) + body.get("max_tokens", 0))

        async def on_response(response):
            await response.aread()
            if response.status_code == 200:
                usage = response.json().get("usage") or {}
                total = usage.get("total_tokens", 0)
                self.tokens += total
                limiter.record(total)
                left = response.headers.get("x-ratelimit-remaining-requests", "")
                if left.isdigit() and int(left) < self.min_requests:
                    self.budget_hit = True
                    self.budget_reset = response.headers.get("x-ratelimit-reset-requests")
            elif response.status_code == 429 and is_daily_limit(response.text):
                self.budget_hit = True
                self.budget_reset = reset_hint(response.text)

        client = AsyncOpenAI(base_url=ev["judge_base_url"], api_key=os.environ["GROQ_API_KEY"],
                             max_retries=ev["judge_max_retries"],
                             http_client=httpx.AsyncClient(timeout=cfg["llm"]["timeout_s"],
                                                           event_hooks={"request": [on_request],
                                                                        "response": [on_response]}))
        llm = llm_factory(judge_model, client=client, temperature=ev["judge_temperature"],
                          max_tokens=ev["judge_max_tokens"], reasoning_effort=ev["judge_reasoning_effort"])
        model = embedder or get_embedder(cfg["embedding"]["model"])

        class BgeEmbeddings(BaseRagasEmbedding):
            def embed_text(self, text: str, **kwargs) -> list[float]:
                return model.embed_query(text)

            async def aembed_text(self, text: str, **kwargs) -> list[float]:
                return model.embed_query(text)

        embeddings = BgeEmbeddings()
        self.metrics = {
            "faithfulness": getattr(collections, METRIC_CLASSES["faithfulness"])(llm=llm),
            "answer_correctness": getattr(collections, METRIC_CLASSES["answer_correctness"])(llm=llm, embeddings=embeddings),
            "answer_relevancy": getattr(collections, METRIC_CLASSES["answer_relevancy"])(llm=llm, embeddings=embeddings),
            "context_precision": getattr(collections, METRIC_CLASSES["context_precision"])(llm=llm),
        }

    def _inputs(self, sample: Sample, metric: str) -> dict:
        inputs = {"user_input": sample.question}
        if metric in ("faithfulness", "answer_correctness", "answer_relevancy"):
            inputs["response"] = sample.answer
        if metric in REFERENCE_METRICS:
            inputs["reference"] = sample.reference
        if metric in CONTEXT_METRICS:
            inputs["retrieved_contexts"] = sample.contexts
        return inputs

    def score(self, sample: Sample, metrics: list[str]) -> dict[str, float | None]:
        out = {}
        for metric in metrics:
            if self.budget_hit:
                raise BudgetExhausted(self.budget_reset)
            try:
                result = self.loop.run_until_complete(self.metrics[metric].ascore(**self._inputs(sample, metric)))
            except Exception:
                if self.budget_hit:
                    raise BudgetExhausted(self.budget_reset)
                raise
            out[metric] = float(result.value)
        return out


class FakeScorer:
    judge_model = "fake-judge"
    version = "fake"

    def __init__(self, scores: dict | None = None, errors: dict | None = None, default: float = 0.5):
        self.scores = scores or {}
        self.errors = errors or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.tokens = 0

    def score(self, sample: Sample, metrics: list[str]) -> dict[str, float | None]:
        out = {}
        for metric in metrics:
            self.calls.append((sample.qid, metric))
            error = self.errors.get((sample.qid, metric)) or self.errors.get(sample.qid)
            if error:
                raise error
            self.tokens += 100
            value = self.scores.get(sample.qid, self.default)
            out[metric] = value[metric] if isinstance(value, dict) else value
        return out
