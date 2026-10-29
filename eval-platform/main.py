"""Command-line entry point.

  python main.py list                                  models and categories
  python main.py validate                              check the test data files
  python main.py run --dry-run                         plan + cost estimate, no calls
  python main.py run [--models a,b] [--limit 1] [--budget 3]     live benchmark (paid!)
  python main.py run --mock                            same pipeline with a fake client (free)
  python main.py rescore                               re-run automatic checks on saved answers (free)
  python main.py report [--narrative]                  Markdown report from saved results
  python main.py dashboard                             interactive HTML dashboard from saved results
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import config
import dashboard
import recommend as rec_mod
import report as report_mod
import runner
import scoring
import testdata
from budget import Budget
from cache import ResponseCache
from llm import LLM


def setup_logging(verbose):
    """Structured key=value log lines go to a file; --verbose also shows them on screen."""
    handlers = [logging.FileHandler(config.LOG_FILE, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")


def make_client(mock, tests):
    if mock:
        from mock_client import MockBedrockRuntime
        return MockBedrockRuntime(tests)
    import boto3
    from botocore.config import Config
    # max_attempts=1: boto3 must not retry silently; our own loop in llm.py does it and logs it.
    return boto3.client("bedrock-runtime", region_name=config.REGION,
                        config=Config(retries={"max_attempts": 1}, read_timeout=120, max_pool_connections=32))


def split_csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def weights_from_args(args):
    weights = dict(config.WEIGHTS)
    for key in ("quality", "cost", "latency"):
        override = getattr(args, f"w_{key}", None)
        if override is not None:
            weights[key] = override
    return weights


def judge_names_from_args(args):
    if getattr(args, "no_judge", False):
        return []
    names = [args.judge or config.JUDGE_MODEL]
    second = args.judge2 if args.judge2 is not None else config.SECOND_JUDGE_MODEL
    if second and second != "none":
        names.append(second)
    for n in names:
        if n not in config.MODELS:
            raise SystemExit(f"unknown judge model '{n}'; choose from {list(config.MODELS)}")
    return names


# ---------- commands -----------------------------------------------------------------------

def cmd_list(args):
    categories = testdata.load_categories()
    print("Models (prices are ESTIMATES in USD per 1M tokens, input/output):")
    for name, spec in config.MODELS.items():
        flag = "default" if spec["default"] else "opt-in "
        print(f"  {name:12s} {flag}  {spec['label']:20s} ${spec['price_in']}/${spec['price_out']}  {spec['id']}")
    print(f"\nCategories ({len(categories)}, {len(testdata.all_tests(categories))} tests):")
    for cat in categories:
        print(f"  {cat['category']:26s} [{cat['group']:10s}] {len(cat['tests'])} tests  {cat['description']}")


def cmd_validate(args):
    categories = testdata.load_categories()
    problems = testdata.validate(categories) + testdata.validate_references(categories)
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(categories)} categories, {len(testdata.all_tests(categories))} tests: "
          + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def write_outputs(out_dir, weights, llm=None, narrative=False):
    """Create report.md and dashboard.html from results.jsonl."""
    records = list(runner.load_results(out_dir).values())
    meta = runner.read_meta(out_dir)
    if not records:
        print("no results found in", out_dir)
        return 1
    text = None
    if narrative and llm:   # optional, paid: an LLM-written paragraph, marked as generated in the report
        summary = scoring.summarize(records, weights)
        text = report_mod.generate_narrative(llm, summary, rec_mod.recommend(summary))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_mod.build_report(records, meta, weights, narrative=text))
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(dashboard.build_dashboard(records, meta, weights))
    print(f"wrote {os.path.join(out_dir, 'report.md')} and {os.path.join(out_dir, 'dashboard.html')}")
    return 0


def cmd_run(args):
    out_dir = args.out or (config.MOCK_OUT_DIR if args.mock else config.DEFAULT_OUT_DIR)
    categories = testdata.load_categories(only=split_csv(args.categories), limit=args.limit)
    if not categories:
        raise SystemExit("no categories selected")
    problems = testdata.validate(categories)
    if problems:
        raise SystemExit("test data problems:\n" + "\n".join(problems))
    models = config.resolve_models(split_csv(args.models))
    judge_names = judge_names_from_args(args)
    if args.no_exec:
        config.ENABLE_CODE_EXEC = False
    tests = testdata.all_tests(categories)
    budget_usd = args.budget if args.budget is not None else config.BUDGET_USD
    cache = None if (args.mock or args.no_cache) else ResponseCache(config.CACHE_DIR)
    done = runner.load_results(out_dir)

    plan = runner.plan_run(models, categories, judge_names, cache, done, use_judge=bool(judge_names))
    print(f"Models ({len(models)}): {', '.join(n for n, _ in models)}")
    print(f"Categories: {len(categories)}, tests: {len(tests)}, judge(s): {', '.join(judge_names) or 'none'}")
    print(f"Planned paid calls: {plan['candidate_calls']} test calls + {plan['judge_calls']} judge calls "
          f"({plan['cached_candidate_calls']} test + {plan['cached_judge_calls']} judge calls served by cache, "
          f"{plan['already_done']} pairs already done in {out_dir})")
    print(f"Estimated cost: ${plan['estimated_usd']:.3f}   budget cap: ${budget_usd:.2f}")
    if args.dry_run:
        for name, usd in plan["per_model_usd"].items():
            print(f"  {name:12s} ~${usd:.3f}")
        print("(dry run: nothing was called. Estimates assume ~%d output tokens per answer.)" % config.EXPECTED_OUTPUT_TOKENS)
        return 0
    if not args.mock and plan["estimated_usd"] > budget_usd:
        print("WARNING: the estimate is above the budget cap; the run will stop when the cap is reached.", file=sys.stderr)

    budget = Budget(budget_usd)
    llm = LLM(make_client(args.mock, tests), cache=cache, budget=budget)
    meta = runner.read_meta(out_dir)
    meta.update({"mock": bool(args.mock), "models": [n for n, _ in models], "judges": judge_names,
                 "budget_usd": budget_usd, "limit": args.limit,
                 "started_at": meta.get("started_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")})
    summary = runner.run_benchmark(llm, models, categories, judge_names, out_dir, workers=args.workers,
                                   use_judge=bool(judge_names))
    meta["spent_usd"] = round(meta.get("spent_usd", 0.0) + summary["spent_usd"], 4)   # cumulative over resumed runs
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["aborted"] = summary["aborted"]
    runner.write_meta(out_dir, meta)

    print(f"\nFinished {summary['finished']}/{summary['pending_at_start']} calls in {summary['wall_seconds']:.1f}s, "
          f"{summary['failed']} failed. Estimated spend this run: ${summary['spent_usd']:.4f}")
    if summary["aborted"]:
        print(f"ABORTED: {summary['abort_reason']}\nResults so far are saved; raise --budget and run again to resume.")
    if not args.no_report:
        write_outputs(out_dir, weights_from_args(args))
    return 3 if summary["aborted"] else 0


def cmd_report(args):
    out_dir = args.out or (config.MOCK_OUT_DIR if args.mock else config.DEFAULT_OUT_DIR)
    llm = None
    if args.narrative:
        llm = LLM(make_client(args.mock, testdata.all_tests(testdata.load_categories())),
                  cache=None if args.mock else ResponseCache(config.CACHE_DIR), budget=Budget(0.25))
    return write_outputs(out_dir, weights_from_args(args), llm=llm, narrative=args.narrative)


def cmd_rescore(args):
    out_dir = args.out or config.DEFAULT_OUT_DIR
    changed = runner.rescore(out_dir, testdata.load_categories())
    print(f"re-ran the automatic checks on saved answers: {changed} record(s) changed")
    return write_outputs(out_dir, weights_from_args(args))


def build_parser():
    p = argparse.ArgumentParser(description="Evaluation platform for Amazon Bedrock models")
    p.add_argument("--verbose", action="store_true", help="show log lines on screen")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show models and test categories").set_defaults(fn=cmd_list)
    sub.add_parser("validate", help="check the test data files").set_defaults(fn=cmd_validate)

    def add_weights(sp):
        sp.add_argument("--w-quality", type=float, help="weight of quality in the aggregate score (default 0.6)")
        sp.add_argument("--w-cost", type=float, help="weight of cost (default 0.2)")
        sp.add_argument("--w-latency", type=float, help="weight of latency (default 0.2)")
        sp.add_argument("--out", help="results folder (default results/, or demo_output/ with --mock)")
        sp.add_argument("--mock", action="store_true", help="use the fake client (no AWS, no cost)")

    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--models", help="comma list of model short names, or 'all' (see: list)")
    r.add_argument("--categories", help="comma list of category names")
    r.add_argument("--limit", type=int, help="max tests per category")
    r.add_argument("--budget", type=float, help=f"hard cap in USD (default {config.BUDGET_USD})")
    r.add_argument("--dry-run", action="store_true", help="print planned calls and estimated cost, call nothing")
    r.add_argument("--judge", help=f"primary judge (default {config.JUDGE_MODEL})")
    r.add_argument("--judge2", help="second judge model, or 'none'")
    r.add_argument("--no-judge", action="store_true", help="automatic checks only")
    r.add_argument("--no-exec", action="store_true", help="do not execute generated code (python/regex checks skipped)")
    r.add_argument("--no-cache", action="store_true", help="ignore the on-disk response cache")
    r.add_argument("--no-report", action="store_true", help="do not write report.md / dashboard.html afterwards")
    r.add_argument("--workers", type=int, help=f"threads (default {config.MAX_WORKERS})")
    add_weights(r)
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="write report.md and dashboard.html from saved results")
    rp.add_argument("--narrative", action="store_true", help="add an LLM-written paragraph (one small paid call)")
    add_weights(rp)
    rp.set_defaults(fn=cmd_report)

    rs = sub.add_parser("rescore", help="re-run automatic checks on saved answers (free), then rebuild outputs")
    add_weights(rs)
    rs.set_defaults(fn=cmd_rescore)

    d = sub.add_parser("dashboard", help="write report.md and dashboard.html from saved results (same as report)")
    add_weights(d)
    d.set_defaults(fn=cmd_report, narrative=False)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
