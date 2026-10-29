"""The only place that talks to Bedrock: Converse call, retry with backoff, cache, budget check."""
import logging
import random
import time
from dataclasses import dataclass

from botocore.exceptions import ClientError

import cache as cache_mod
import config
from budget import Budget

log = logging.getLogger("eval-platform")


def extract_text(response):
    """Join ONLY the content blocks that contain 'text'.

    Why: some models return other block types (e.g. reasoning). Reading content[0]['text']
    blindly crashes with KeyError when the first block is not text.
    """
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(b["text"] for b in blocks if "text" in b).strip()


def estimate_tokens(text):
    return max(1, len(text) // 4)   # rule of thumb: 1 token is about 4 characters


@dataclass
class CallResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    cached: bool = False
    stop_reason: str = ""
    ok: bool = True
    error: str = ""
    error_kind: str = ""    # "api" (Bedrock failed) or "empty" (model returned no text)


class LLM:
    def __init__(self, client, cache=None, budget=None, sleep=time.sleep):
        self.client = client                       # boto3 'bedrock-runtime' client (or a fake)
        self.cache = cache                         # ResponseCache or None
        self.budget = budget or Budget(float("inf"))
        self.sleep = sleep                         # injectable so tests do not really wait

    def call(self, model_id, prompt, system=None, max_tokens=None, temperature=None):
        """Send one prompt. Returns a CallResult. Raises only BudgetExceeded.

        API failures do NOT raise: a benchmark must keep going and record "this model failed here".
        """
        max_tokens = max_tokens or config.DEFAULT_MAX_TOKENS
        temperature = config.TEMPERATURE if temperature is None else temperature

        key = cache_mod.make_key(model_id, system, prompt, max_tokens, temperature)
        if self.cache:
            hit = self.cache.get(key)
            if hit:
                return CallResult(cached=True, **hit)

        # Budget check BEFORE paying: assume the worst case (the model uses its whole output budget).
        worst = config.cost_usd(model_id, estimate_tokens((system or "") + prompt), max_tokens)
        self.budget.check(worst)

        request = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            request["system"] = [{"text": system}]

        start = time.perf_counter()
        response = None
        attempt = 0
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                response = self.client.converse(**request)
                break
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "")
                if code not in config.RETRYABLE_ERROR_CODES or attempt == config.MAX_ATTEMPTS:
                    log.error("call failed model=%s code=%s attempts=%d", config.short_name(model_id), code, attempt)
                    return CallResult(latency_s=time.perf_counter() - start, attempts=attempt, ok=False,
                                      error=f"{code or err}", error_kind="api")
                wait = self._backoff(attempt)
                log.warning("retry model=%s code=%s attempt=%d wait=%.1fs",
                            config.short_name(model_id), code, attempt, wait)
                self.sleep(wait)

        latency = time.perf_counter() - start
        usage = response.get("usage", {})
        tin, tout = usage.get("inputTokens", 0), usage.get("outputTokens", 0)
        self.budget.add(config.cost_usd(model_id, tin, tout))
        text = extract_text(response)
        stop = response.get("stopReason", "")
        if stop == "max_tokens":
            log.warning("truncated model=%s: hit maxTokens=%d", config.short_name(model_id), max_tokens)
        result = CallResult(text=text, input_tokens=tin, output_tokens=tout, latency_s=latency,
                            attempts=attempt, stop_reason=stop)
        if not text:
            result.ok, result.error, result.error_kind = False, f"no text returned (stopReason={stop})", "empty"
            return result   # not cached: a retry may work
        if self.cache:
            self.cache.put(key, {"text": text, "input_tokens": tin, "output_tokens": tout,
                                 "latency_s": latency, "attempts": attempt, "stop_reason": stop})
        return result

    @staticmethod
    def _backoff(attempt):
        """1s, 2s, 4s, 8s ... capped, plus jitter so parallel callers do not retry in lockstep."""
        delay = min(config.BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), config.BACKOFF_MAX_SECONDS)
        return delay + random.uniform(0, delay * 0.25)
