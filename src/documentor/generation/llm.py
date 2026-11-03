import os
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import groq
import tiktoken
from dotenv import load_dotenv

from documentor.config import resolve
from documentor.generation.cache import ResponseCache, cache_key


class GenerationError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning: str | None
    latency_ms: int
    cache_hit: bool
    remaining_tokens: int | None   # from x-ratelimit-remaining-tokens, if present
    reasoning_tokens: int | None = None
    wait_ms: int = 0               # rate-limiter waits and retry sleeps


class LLM(Protocol):
    def complete(self, messages: list[dict], **overrides) -> LLMResult: ...


@lru_cache(maxsize=2)
def get_encoding(name: str):
    return tiktoken.get_encoding(name)


def count_message_tokens(messages: list[dict], encoding: str = "o200k_base") -> int:
    enc = get_encoding(encoding)
    return sum(len(enc.encode(m["content"])) + 4 for m in messages)


class TokenRateLimiter:
    """Sliding 60-second window over tokens used, so calls stay under the TPM limit."""

    def __init__(self, tpm: int, clock=time.monotonic, sleep=time.sleep, window: float = 60.0):
        self.tpm = tpm
        self.clock = clock
        self.sleep = sleep
        self.window = window
        self.events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.events and self.events[0][0] <= now - self.window:
            self.events.popleft()

    def used(self) -> int:
        self._prune(self.clock())
        return sum(t for _, t in self.events)

    def wait(self, estimate: int) -> None:
        while True:
            now = self.clock()
            self._prune(now)
            if not self.events or self.used() + estimate <= self.tpm:
                return
            delay = self.events[0][0] + self.window - now
            if delay > 2:
                print(f"rate limiter: waiting {delay:.0f}s to stay under {self.tpm:,} tokens/min")
            self.sleep(delay)

    def record(self, tokens: int) -> None:
        self.events.append((self.clock(), tokens))


def backoff(attempt: int) -> float:
    return 2 ** attempt + random.uniform(0, 0.5)


def retry_after(err: groq.APIStatusError) -> float | None:
    value = err.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class GroqLLM:
    def __init__(self, cfg: dict, client=None, cache: ResponseCache | None = None,
                 limiter: TokenRateLimiter | None = None, sleep=time.sleep):
        self.cfg = cfg["llm"]
        self.encoding = cfg["generation"]["budget_encoding"]
        if client is None:
            load_dotenv(resolve(".env"))
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")
            client = groq.Groq(api_key=key, timeout=self.cfg["timeout_s"], max_retries=0)
        self.client = client
        if cache is None and self.cfg.get("cache"):
            cache = ResponseCache(resolve(self.cfg["cache_path"]))
        self.cache = cache
        self.limiter = limiter or TokenRateLimiter(self.cfg["tpm_limit"], sleep=sleep)
        self.sleep = sleep

    def complete(self, messages: list[dict], **overrides) -> LLMResult:
        show_reasoning = overrides.pop("show_reasoning", False)
        use_cache = overrides.pop("use_cache", True)
        params = {
            "model": self.cfg["model"],
            "temperature": self.cfg["temperature"],
            "reasoning_effort": self.cfg["reasoning_effort"],
            "max_completion_tokens": self.cfg["max_completion_tokens"],
            "include_reasoning": show_reasoning,
        } | overrides

        key = cache_key(params, messages)
        if self.cache and use_cache:
            hit = self.cache.get(key)
            if hit:
                return hit

        wait_start = time.perf_counter()
        self.limiter.wait(count_message_tokens(messages, self.encoding) + params["max_completion_tokens"])
        limiter_ms = int((time.perf_counter() - wait_start) * 1000)
        result = self._call_with_retries(messages, params)
        result.wait_ms += limiter_ms
        self.limiter.record(result.prompt_tokens + result.completion_tokens)
        if not result.text and result.finish_reason == "length":
            raise GenerationError("completion budget used up by reasoning; "
                                  "raise max_completion_tokens or lower reasoning_effort")
        if self.cache:
            self.cache.put(key, result)
        return result

    def _call_with_retries(self, messages: list[dict], params: dict) -> LLMResult:
        attempt = 0
        slept = 0.0
        while True:
            try:
                result = self._call(messages, params)
                result.wait_ms = int(slept * 1000)
                return result
            except groq.RateLimitError as e:
                err, delay = e, retry_after(e) or backoff(attempt)
            except (groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError) as e:
                err, delay = e, backoff(attempt)
            except groq.APIStatusError as e:
                msg = str(e).lower()
                if "model_decommissioned" in msg or (e.status_code == 404 and "model" in msg):
                    print("model retired or renamed — update llm.model in config")
                raise
            attempt += 1
            if attempt > self.cfg["max_retries"]:
                raise err
            sleep_start = time.perf_counter()
            self.sleep(delay)
            slept += time.perf_counter() - sleep_start

    def _call(self, messages: list[dict], params: dict) -> LLMResult:
        start = time.perf_counter()
        raw = self.client.chat.completions.with_raw_response.create(
            messages=messages, stream=False, **params)
        resp = raw.parse()
        latency = int((time.perf_counter() - start) * 1000)
        choice = resp.choices[0]
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        remaining = raw.headers.get("x-ratelimit-remaining-tokens")
        return LLMResult(
            text=choice.message.content or "",
            model=resp.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning=getattr(choice.message, "reasoning", None),
            latency_ms=latency,
            cache_hit=False,
            remaining_tokens=int(remaining) if remaining and remaining.isdigit() else None,
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
        )


class FakeLLM:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **overrides) -> LLMResult:
        self.calls.append(messages)
        item = self.responses.pop(0)
        if isinstance(item, str):
            item = LLMResult(text=item, model="fake", finish_reason="stop", prompt_tokens=100,
                             completion_tokens=20, reasoning=None, latency_ms=1, cache_hit=False,
                             remaining_tokens=None)
        return item


def get_llm(cfg: dict) -> LLM:
    provider = cfg["llm"]["provider"]
    if provider == "groq":
        return GroqLLM(cfg)
    raise ValueError(f"unknown llm provider: {provider}")
