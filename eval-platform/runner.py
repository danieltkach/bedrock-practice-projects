"""Run the benchmark: every (model, test) pair, concurrently, saving each result as it completes.

Flow for one pair (`evaluate_one`):  ask the model -> automatic checks -> LLM-judge (if the test wants it)
-> one JSON line appended to results.jsonl.  Because lines are saved immediately, a crashed or aborted
run can be resumed: finished pairs are skipped next time.
"""
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cache as cache_mod
import checks
import config
import judge as judge_mod
from budget import BudgetExceeded
from llm import estimate_tokens

log = logging.getLogger("eval-platform")


# ---------- results file -------------------------------------------------------------------

def results_path(out_dir):
    return os.path.join(out_dir, "results.jsonl")


def load_results(out_dir):
    """Read results.jsonl -> {(model, test_id): record}. The LAST line per pair wins (a retry overrides a failure)."""
    records = {}
    try:
        with open(results_path(out_dir), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records[(rec["model"], rec["test_id"])] = rec
    except FileNotFoundError:
        pass
    return records


def test_fingerprint(test):
    """Short hash of what the MODEL sees. If a test's prompt/system/token budget changes, old results are stale."""
    payload = json.dumps([test["prompt"], test.get("system"), test.get("max_tokens")])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def is_done(done, model_name, test):
    """True if a successful, up-to-date result for this pair already exists (resume support)."""
    rec = done.get((model_name, test["id"]))
    return bool(rec and rec["ok"] and rec.get("fp") == test_fingerprint(test))


# ---------- one (model, test) pair ----------------------------------------------------------

def evaluate_one(llm, model_name, spec, test, judge_names, use_judge=True):
    res = llm.call(spec["id"], test["prompt"], system=test.get("system"), max_tokens=test.get("max_tokens"))
    rec = {
        "model": model_name, "model_id": spec["id"], "category": test["category"], "group": test["group"],
        "test_id": test["id"], "fp": test_fingerprint(test), "response": res.text, "ok": res.ok, "error": res.error,
        "error_kind": res.error_kind, "stop_reason": res.stop_reason,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "latency_s": round(res.latency_s, 3), "cached": res.cached, "attempts": res.attempts,
        "auto": None, "judge": None, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not res.ok:
        return rec
    if test.get("checks"):
        rec["auto"] = checks.run_checks(test["checks"], res.text)
    if use_judge and test.get("judge", True) and judge_names:
        scores = {}
        for judge_name in judge_mod.judges_for(model_name, judge_names):
            scores[judge_name] = judge_mod.judge_answer(llm, config.MODELS[judge_name]["id"], test, res.text)
        mean, used, self_judged = judge_mod.combine_judges(scores, model_name)
        rec["judge"] = {"scores": scores, "mean": mean, "used": used, "self_judged": self_judged}
    return rec


# ---------- planning (used by --dry-run) ---------------------------------------------------

def plan_run(models, categories, judge_names, cache, done, use_judge=True):
    """Count the calls a run would make and estimate their cost, without calling anything."""
    summary = {"candidate_calls": 0, "judge_calls": 0, "already_done": 0, "cached_candidate_calls": 0, "cached_judge_calls": 0,
               "estimated_usd": 0.0, "per_model_usd": {}}
    for name, spec in models:
        model_cost = 0.0
        for cat in categories:
            for test in cat["tests"]:
                if is_done(done, name, test):
                    summary["already_done"] += 1
                    continue
                max_tokens = test.get("max_tokens") or config.DEFAULT_MAX_TOKENS
                key = cache_mod.make_key(spec["id"], test.get("system"), test["prompt"], max_tokens, config.TEMPERATURE)
                hit = cache.get(key) if cache else None
                if hit:
                    summary["cached_candidate_calls"] += 1   # free
                else:
                    summary["candidate_calls"] += 1
                    tout = min(config.EXPECTED_OUTPUT_TOKENS, max_tokens)
                    model_cost += config.cost_usd(spec["id"], estimate_tokens(test["prompt"]), tout)
                if use_judge and test.get("judge", True):
                    for judge_name in judge_mod.judges_for(name, judge_names):
                        judge_id = config.MODELS[judge_name]["id"]
                        if hit:   # we know the answer text, so we can tell whether this judge call is cached too
                            jkey = cache_mod.make_key(judge_id, judge_mod.JUDGE_SYSTEM,
                                                      judge_mod.build_judge_prompt(test, hit["text"]),
                                                      config.JUDGE_MAX_TOKENS, 0.0)
                            if cache.get(jkey):
                                summary["cached_judge_calls"] += 1
                                continue
                        summary["judge_calls"] += 1
                        tin = (config.JUDGE_EXPECTED_INPUT_TOKENS + estimate_tokens(test["prompt"])
                               + estimate_tokens(test.get("reference", "")) + config.EXPECTED_OUTPUT_TOKENS)
                        model_cost += config.cost_usd(judge_id, tin, 60)
        summary["per_model_usd"][name] = model_cost
        summary["estimated_usd"] += model_cost
    return summary


# ---------- the concurrent run --------------------------------------------------------------

def run_benchmark(llm, models, categories, judge_names, out_dir, workers=None, use_judge=True, progress=True):
    """Run all pending (model, test) pairs. Returns a summary dict. Never raises BudgetExceeded: it reports it."""
    workers = workers or config.MAX_WORKERS
    os.makedirs(out_dir, exist_ok=True)
    done = load_results(out_dir)
    # Interleave: test by test, all models for each test, so an aborted run still compares models fairly.
    pending = [(name, spec, test) for cat in categories for test in cat["tests"] for name, spec in models
               if not is_done(done, name, test)]
    total = len(pending)
    state = {"finished": 0, "failed": 0, "stop": False, "reason": ""}
    lock = threading.Lock()
    semaphores = {name: threading.Semaphore(config.PER_MODEL_CONCURRENCY) for name, _ in models}
    started = time.perf_counter()

    out_file = open(results_path(out_dir), "a", encoding="utf-8")

    def work(item):
        name, spec, test = item
        if state["stop"]:
            return
        with semaphores[name]:   # at most N simultaneous calls to one model: friendlier to rate limits
            if state["stop"]:
                return
            try:
                rec = evaluate_one(llm, name, spec, test, judge_names, use_judge)
            except BudgetExceeded as err:
                state["stop"], state["reason"] = True, str(err)
                log.error("aborting: %s", err)
                return
            except Exception as err:   # a bug in one pair must not kill the whole run
                log.exception("unexpected error model=%s test=%s", name, test["id"])
                rec = {"model": name, "model_id": spec["id"], "category": test["category"], "group": test["group"],
                       "test_id": test["id"], "fp": test_fingerprint(test), "response": "", "ok": False, "error": f"internal: {err}",
                       "error_kind": "api", "stop_reason": "", "input_tokens": 0, "output_tokens": 0,
                       "latency_s": 0, "cached": False, "attempts": 0, "auto": None, "judge": None, "ts": ""}
        with lock:   # one writer at a time keeps each JSON line intact
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_file.flush()
            state["finished"] += 1
            state["failed"] += 0 if rec["ok"] else 1
            if progress:
                tag = "cached" if rec["cached"] else f"{rec['latency_s']:.1f}s"
                status = "ok" if rec["ok"] else "FAILED"
                print(f"[{state['finished']}/{total}] {name:12s} {test['id']:34s} {status} {tag}",
                      file=sys.stderr, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, pending))
    finally:
        out_file.close()

    summary = {"pending_at_start": total, "finished": state["finished"], "failed": state["failed"],
               "skipped_already_done": len(done), "aborted": state["stop"], "abort_reason": state["reason"],
               "spent_usd": llm.budget.spent, "wall_seconds": time.perf_counter() - started}
    log.info("run summary: %s", summary)
    return summary


def write_meta(out_dir, meta):
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def read_meta(out_dir):
    try:
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def rescore(out_dir, categories):
    """Re-run the AUTOMATIC checks on saved answers (free, no model calls).

    Use it after fixing a check in a test file. Judge scores are kept as they are; if you changed a
    rubric, delete the results instead. Returns how many records changed.
    """
    tests = {t["id"]: t for c in categories for t in c["tests"]}
    records = load_results(out_dir)
    changed = 0
    for rec in records.values():
        test = tests.get(rec["test_id"])
        if not test or not rec["ok"] or not test.get("checks"):
            continue
        new_auto = checks.run_checks(test["checks"], rec["response"])
        if new_auto != rec.get("auto"):
            rec["auto"] = new_auto
            changed += 1
    with open(results_path(out_dir), "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed
