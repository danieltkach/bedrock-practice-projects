"""One place that talks to Bedrock: Converse call, retry with backoff, token/latency tracking."""
import logging
import random
import time

from botocore.exceptions import ClientError

import config
from metrics import CallRecord

log = logging.getLogger("multi-agent")


class LLMError(Exception):
    """Raised when a call fails for good (after retries) or returns no text."""


def extract_text(response):
    """Join ONLY the content blocks that contain 'text'.

    Why: some models return other block types (e.g. reasoning). Reading content[0]['text']
    blindly crashes with KeyError when the first block is not text.
    """
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(b["text"] for b in blocks if "text" in b).strip()


class LLM:
    def __init__(self, client, tracker, sleep=time.sleep):
        self.client = client      # a boto3 'bedrock-runtime' client (or a fake one in tests)
        self.tracker = tracker
        self.sleep = sleep        # injectable so tests do not really wait

    def call(self, role, system, prompt, model_id=None, max_tokens=None, temperature=0.2):
        """Send one prompt, return the text. Retries throttling; always records usage."""
        model_id = model_id or config.ROLE_MODEL[role]
        max_tokens = max_tokens or config.ROLE_MAX_TOKENS[role]
        request = {
            "modelId": model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        start = time.perf_counter()
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                response = self.client.converse(**request)
                break
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "")
                if code not in config.RETRYABLE_ERROR_CODES or attempt == config.MAX_ATTEMPTS:
                    self._record(role, model_id, 0, 0, start, attempt, ok=False)
                    raise LLMError(f"{role}: Bedrock error {code or err}") from err
                wait = self._backoff(attempt)
                log.warning("retry role=%s code=%s attempt=%d wait=%.1fs", role, code, attempt, wait)
                self.sleep(wait)

        usage = response.get("usage", {})
        text = extract_text(response)
        ok = bool(text)
        if response.get("stopReason") == "max_tokens":
            # The answer was cut off. Not fatal, but worth knowing: raise ROLE_MAX_TOKENS.
            log.warning("truncated role=%s: hit maxTokens=%d", role, max_tokens)
        self._record(role, model_id, usage.get("inputTokens", 0), usage.get("outputTokens", 0),
                     start, attempt, ok=ok)
        if not ok:
            raise LLMError(f"{role}: no text returned (stopReason={response.get('stopReason')})")
        return text

    @staticmethod
    def _backoff(attempt):
        """1s, 2s, 4s, 8s ... capped, plus jitter so parallel callers do not retry in lockstep."""
        delay = min(config.BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), config.BACKOFF_MAX_SECONDS)
        return delay + random.uniform(0, delay * 0.25)

    def _record(self, role, model_id, tin, tout, start, attempts, ok):
        elapsed = time.perf_counter() - start
        self.tracker.add(CallRecord(role, model_id, tin, tout, elapsed, attempts, ok))
        log.info("call role=%s model=%s in=%d out=%d sec=%.2f attempts=%d ok=%s",
                 role, model_id, tin, tout, elapsed, attempts, ok)
