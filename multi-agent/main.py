"""Command-line entry point.

  python main.py "your question"       single-shot
  python main.py                       interactive chat
  python main.py --mock "..."          no AWS needed (fake client)
  python main.py --examples            list the example queries
"""
import argparse
import logging
import sys

import config
from llm import LLM
from metrics import Tracker
from orchestrator import answer_query, describe_plan

EXAMPLES = [
    "Hi! What can you do?",                                                         # direct
    "Write a Python function that removes duplicates from a list but keeps the order.",  # single (code)
    "Compare DynamoDB and Aurora for a shopping-cart service and recommend one.",    # single (analysis)
    "Compare SQS and Kinesis for processing 5,000 events per second, write a Python "
    "consumer example for the better one, and draft a short email to my team explaining the choice.",  # multi
    "Analyze the pros and cons of moving a monolith to microservices, then write a "
    "one-paragraph executive summary for non-technical managers.",                    # multi with dependency
]


def setup_logging(verbose):
    """Routing decisions and call stats go to a log file; --verbose also shows them on screen."""
    handlers = [logging.FileHandler(config.LOG_FILE, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)


def make_client(mock):
    if mock:
        from mock_client import MockBedrockRuntime
        return MockBedrockRuntime()
    import boto3
    from botocore.config import Config
    # max_attempts=1: boto3 must not retry silently; our own loop in llm.py does it and logs it.
    return boto3.client("bedrock-runtime", region_name=config.REGION,
                        config=Config(retries={"max_attempts": 1}, read_timeout=120))


def run_once(llm, tracker, query, history_text="", show_report=True):
    tracker.reset()
    result = answer_query(llm, query, history_text)
    print(f"\n[routing] {describe_plan(result.plan)}")
    if result.plan.reasoning:
        print(f"[why] {result.plan.reasoning}")
    print(f"\n{result.answer}")
    if show_report:
        print(tracker.report(result.wall_seconds))
    return result


def format_history(history):
    recent = history[-config.HISTORY_TURNS:]
    return "\n".join(f"Q: {q}\nA: {a[:config.HISTORY_CHARS]}" for q, a in recent)


def chat(llm, tracker, show_report):
    print("Multi-agent assistant. Commands: /examples  /quit")
    history = []
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in ("/quit", "/exit"):
            break
        if query == "/examples":
            print("\n".join(f"{i}. {e}" for i, e in enumerate(EXAMPLES, 1)))
            continue
        result = run_once(llm, tracker, query, format_history(history), show_report)
        history.append((query, result.answer))


def main():
    parser = argparse.ArgumentParser(description="Multi-agent system on Amazon Bedrock")
    parser.add_argument("query", nargs="?", help="question; omit for interactive chat")
    parser.add_argument("--mock", action="store_true", help="use a fake client (no AWS needed)")
    parser.add_argument("--verbose", action="store_true", help="show log lines on screen")
    parser.add_argument("--no-report", action="store_true", help="hide the cost/latency report")
    parser.add_argument("--examples", action="store_true", help="print example queries and exit")
    args = parser.parse_args()

    if args.examples:
        print("\n".join(f"{i}. {e}" for i, e in enumerate(EXAMPLES, 1)))
        return
    setup_logging(args.verbose)
    tracker = Tracker()
    llm = LLM(make_client(args.mock), tracker)
    if args.query:
        run_once(llm, tracker, args.query, show_report=not args.no_report)
    else:
        chat(llm, tracker, not args.no_report)


if __name__ == "__main__":
    main()
