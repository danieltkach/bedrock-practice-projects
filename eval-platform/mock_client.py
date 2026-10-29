"""A fake 'bedrock-runtime' client that needs no AWS. Used by --mock mode and by the tests.

How it fakes an evaluation: it knows every test's reference answer. Each fake model has a "skill"
(chance of returning the reference answer; otherwise a vague wrong answer) and a speed. The choice is
deterministic (a hash of model + prompt), so demo runs are reproducible. The fake judge compares the
candidate with the reference. The numbers it produces are NOT real model quality: it only exercises
the pipeline (calls, checks, judging, scoring, report, dashboard).
"""
import hashlib
import json
import re
import time

# model id -> (skill 0..1, seconds of fake network time)
PROFILES = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (0.95, 0.030),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.88, 0.012),
    "us.amazon.nova-pro-v1:0": (0.82, 0.020),
    "us.amazon.nova-2-lite-v1:0": (0.78, 0.018),
    "us.amazon.nova-lite-v1:0": (0.70, 0.008),
    "us.amazon.nova-micro-v1:0": (0.58, 0.005),
    "us.anthropic.claude-opus-4-6-v1": (0.97, 0.040),
}
WRONG_ANSWER = "I am not completely sure, but it is probably something like that."


def _tokens(text):
    return max(1, len(text) // 4)


def _unit_hash(*parts):
    """A repeatable pseudo-random number in [0, 1) derived from the inputs."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


class MockBedrockRuntime:
    def __init__(self, tests, speed_factor=1.0):
        self.references = {t["prompt"]: t.get("reference", "") for t in tests}
        self.speed_factor = speed_factor

    def converse(self, **request):
        model = request["modelId"]
        system = request.get("system", [{"text": ""}])[0]["text"]
        prompt = request["messages"][0]["content"][0]["text"]
        skill, delay = PROFILES.get(model, (0.7, 0.01))
        time.sleep(delay * self.speed_factor)
        if system.startswith("You are a strict, impartial evaluator"):
            text = self._judge(prompt)
        elif "NARRATIVE WRITER" in system:   # the optional narrative call
            text = "(mock narrative) The top model leads on quality; the cheapest model wins on value."
        else:
            reference = self.references.get(prompt, "")
            good = _unit_hash(model, prompt) < skill and reference
            text = reference if good else WRONG_ANSWER
        return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": _tokens(system + prompt), "outputTokens": _tokens(text) + 5}}

    @staticmethod
    def _judge(prompt):
        candidate = prompt.split("CANDIDATE ANSWER:\n")[-1].strip()
        match = re.search(r"REFERENCE ANSWER:\n(.*?)\n\nRUBRIC:", prompt, flags=re.DOTALL)
        reference = match.group(1).strip() if match else ""
        score = 5 if candidate == reference else 2
        return json.dumps({"score": score, "reason": "(mock judge) matches the reference" if score == 5
                           else "(mock judge) does not match the reference"})
