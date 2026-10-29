"""Live end-to-end check against real Bedrock. Run via ./run.sh test-live (needs valid credentials).

It makes a handful of small calls (a few cents at most) and asserts the pipeline works:
direct route, single route, multi route with a parallel wave, and the cost report.
"""
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

import config
from llm import LLM
from main import EXAMPLES, make_client
from metrics import Tracker
from orchestrator import answer_query, describe_plan


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

    tracker = Tracker()
    llm = LLM(make_client(mock=False), tracker)
    ok = True
    # (example index, expected route)
    for index, expected in ((0, "direct"), (1, "single"), (3, "multi")):
        tracker.reset()
        query = EXAMPLES[index]
        print(f"\nQuery: {query[:70]}...")
        result = answer_query(llm, query)
        print(f"  plan: {describe_plan(result.plan)}")
        ok &= check(f"route is {expected} (model-dependent, informational)", result.plan.route == expected)
        ok &= check("answer is not empty", len(result.answer) > 20)
        ok &= check("no fallback plan", not result.plan.fallback)
        ok &= check("calls were tracked", len(tracker.records) >= 1)
        print(tracker.report(result.wall_seconds))
    print("\nLIVE TEST " + ("PASSED" if ok else "FINISHED WITH FAILURES (read the FAIL lines)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
