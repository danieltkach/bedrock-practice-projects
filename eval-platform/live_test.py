"""Live end-to-end check against real Bedrock. Run via ./run.sh test-live (needs valid credentials).

It makes a handful of tiny calls (a few cents at most): one per default model, plus one judge call,
and runs two real tests through the whole pipeline (answer -> checks -> judge -> scoring).
"""
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

import config
import runner
import scoring
import testdata
from budget import Budget
from judge import judge_answer
from llm import LLM
from main import make_client


def check(name, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def main():
    try:
        who = boto3.client("sts", region_name=config.REGION).get_caller_identity()
        print(f"Credentials OK (account {who['Account']}), region {config.REGION}")
    except (ClientError, NoCredentialsError, Exception) as err:
        print(f"No valid credentials: {err}\nRun: aws login --remote --region us-east-1")
        return 2

    llm = LLM(make_client(mock=False, tests=[]), budget=Budget(0.25))
    ok = True
    print("\n1. Every default model answers a tiny prompt with text:")
    for name, spec in config.resolve_models(None):
        res = llm.call(spec["id"], "Reply with the single word: OK", max_tokens=200)
        ok &= check(f"{name}: text returned ({res.latency_s:.1f}s, {res.input_tokens}+{res.output_tokens} tokens)", res.ok and len(res.text) > 0)

    print("\n2. The judge returns parseable JSON:")
    test = {"prompt": "What is 2+2?", "rubric": "Correct answer is 4.", "reference": "4"}
    verdict = judge_answer(llm, config.MODELS[config.JUDGE_MODEL]["id"], test, "4")
    ok &= check(f"judge score for a correct answer is 4-5 (got {verdict['score']})", verdict["score"] in (4, 5))

    print("\n3. Two real tests through the full pipeline (haiku-4.5):")
    cats = testdata.load_categories(only=["arithmetic"], limit=1)
    test = testdata.all_tests(cats)[0]
    rec = runner.evaluate_one(llm, "haiku-4.5", config.MODELS["haiku-4.5"], test, [config.JUDGE_MODEL])
    ok &= check("answer recorded", rec["ok"])
    ok &= check("automatic check ran", rec["auto"] is not None)
    ok &= check("quality computed", scoring.quality_of(rec) is not None)
    print(f"  quality={scoring.quality_of(rec)}, estimated spend so far ${llm.budget.spent:.4f}")
    print("\nLIVE TEST " + ("PASSED" if ok else "FINISHED WITH FAILURES (read the FAIL lines)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
