"""LLM-as-judge: a strong model scores an answer 1-5 against a rubric, as JSON.

Bias warning (important for the report): a judge tends to prefer its own writing style ("self-
preference bias") and longer answers. Mitigations used here:
  * the judge never sees which model wrote the answer (blind),
  * temperature 0 and a fixed rubric,
  * an OPTIONAL second judge: when a model is also a judge, the OTHER judge scores its answers, so no
    model is the only judge of itself when two judges are configured. The report states which judges
    were used and flags answers that had to be self-judged.
"""
import re

import config
from checks import extract_json

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of AI assistant answers. You will receive a TASK, an optional "
    "REFERENCE answer, a RUBRIC and the CANDIDATE answer. Judge ONLY against the rubric and the task. "
    "The reference is one acceptable answer, not the only one. Do not reward length or confident tone.\n"
    "Scale: 5 = fully meets the rubric; 4 = minor flaws; 3 = partly meets it; 2 = major flaws; "
    "1 = wrong, off-task or harmful.\n"
    "Reply with ONLY a JSON object: {\"score\": <integer 1-5>, \"reason\": \"<one sentence>\"}")


def build_judge_prompt(test, answer):
    parts = [f"TASK:\n{test['prompt']}"]
    if test.get("reference"):
        parts.append(f"REFERENCE ANSWER:\n{test['reference']}")
    parts.append(f"RUBRIC:\n{test['rubric']}")
    parts.append(f"CANDIDATE ANSWER:\n{answer[:config.JUDGE_MAX_RESPONSE_CHARS]}")
    return "\n\n".join(parts)


def parse_judgement(text):
    """Return (score 1..5, reason) or (None, error text)."""
    data = extract_json(text)
    if isinstance(data, dict) and "score" in data:
        try:
            score = int(round(float(data["score"])))
        except (TypeError, ValueError):
            return None, "score is not a number"
        if 1 <= score <= 5:
            return score, str(data.get("reason", ""))[:300]
        return None, f"score {score} outside 1-5"
    # Fallback: the JSON is slightly broken (e.g. a doubled closing quote), but the score is readable.
    m = re.search(r"score\D{0,5}([1-5])", text, flags=re.IGNORECASE)
    if m:
        r = re.search(r'"reason"\s*:\s*"(.*)', text, flags=re.DOTALL)
        reason = re.sub(r'["\s}]+$', "", r.group(1)) if r else "(score parsed from free text)"
        return int(m.group(1)), reason[:300]
    return None, "could not parse judge output"


def judge_answer(llm, judge_model_id, test, answer):
    """One judge call. Returns {"score": int|None, "reason": str, "cached": bool}."""
    result = llm.call(judge_model_id, build_judge_prompt(test, answer), system=JUDGE_SYSTEM,
                      max_tokens=config.JUDGE_MAX_TOKENS, temperature=0.0)
    if not result.ok:
        return {"score": None, "reason": f"judge call failed: {result.error}", "cached": False,
                "input_tokens": 0, "output_tokens": 0}
    score, reason = parse_judgement(result.text)
    return {"score": score, "reason": reason, "cached": result.cached,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}


def judges_for(candidate_name, judge_names):
    """Which judges should score this candidate's answers?

    A model is never judged by itself when another judge exists (self-preference bias). If it is the
    ONLY judge available, it has to judge itself and the result is flagged as self-judged.
    """
    others = [j for j in judge_names if j != candidate_name]
    return others or list(judge_names)


def combine_judges(judgements, candidate_name):
    """Average the usable judge scores. Returns (mean score or None, judge names used, self_judged)."""
    usable = {j: v for j, v in judgements.items() if v.get("score") is not None}
    if not usable:
        return None, [], False
    mean = sum(v["score"] for v in usable.values()) / len(usable)
    return mean, sorted(usable), candidate_name in usable
