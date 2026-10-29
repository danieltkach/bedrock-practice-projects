"""Offline tests: no AWS, no network. Run with: ./run.sh test   (or python -m unittest discover)"""
import contextlib
import io
import json
import os
import re
import tempfile
import threading
import time
import unittest

from botocore.exceptions import ClientError

import cache as cache_mod
import checks
import config
import dashboard
import judge
import main as main_mod
import recommend
import report
import runner
import sandbox
import scoring
import testdata
from budget import Budget, BudgetExceeded
from llm import LLM, extract_text
from mock_client import MockBedrockRuntime

SONNET = config.MODELS["sonnet-4.5"]["id"]
HAIKU = config.MODELS["haiku-4.5"]["id"]


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")


def reply(text, tin=10, tout=5):
    return {"output": {"message": {"content": [{"text": text}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": tin, "outputTokens": tout}}


class ScriptedClient:
    """Returns/raises the queued items in order. Records every request."""
    def __init__(self, items):
        self.items, self.requests = list(items), []

    def converse(self, **request):
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(client, **kw):
    return LLM(client, sleep=lambda s: None, **kw)   # no real waiting


class LLMTests(unittest.TestCase):
    def test_extract_text_skips_non_text_blocks(self):
        resp = {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": "hi"}]}}}
        self.assertEqual(extract_text(resp), "hi")

    def test_reasoning_only_response_is_an_empty_failure_and_not_cached(self):
        client = ScriptedClient([{"output": {"message": {"content": [{"reasoningContent": {}}]}},
                                  "stopReason": "max_tokens", "usage": {"inputTokens": 5, "outputTokens": 99}}])
        with tempfile.TemporaryDirectory() as d:
            llm = make_llm(client, cache=cache_mod.ResponseCache(d))
            res = llm.call(SONNET, "p")
            self.assertFalse(res.ok)
            self.assertEqual(res.error_kind, "empty")
            self.assertIsNone(llm.cache.get(cache_mod.make_key(SONNET, None, "p", config.DEFAULT_MAX_TOKENS, config.TEMPERATURE)))

    def test_retries_throttling_then_succeeds(self):
        client = ScriptedClient([client_error("ThrottlingException")] * 2 + [reply("ok")])
        res = make_llm(client).call(SONNET, "p")
        self.assertEqual((res.text, res.attempts, res.ok), ("ok", 3, True))

    def test_gives_up_after_max_attempts_without_raising(self):
        client = ScriptedClient([client_error("ThrottlingException")] * config.MAX_ATTEMPTS)
        res = make_llm(client).call(SONNET, "p")
        self.assertFalse(res.ok)
        self.assertEqual(res.error_kind, "api")
        self.assertEqual(len(client.requests), config.MAX_ATTEMPTS)

    def test_non_retryable_error_fails_immediately(self):
        client = ScriptedClient([client_error("AccessDeniedException")])
        res = make_llm(client).call(SONNET, "p")
        self.assertEqual((res.ok, res.attempts), (False, 1))

    def test_request_shape_system_is_optional(self):
        client = ScriptedClient([reply("a"), reply("b")])
        llm = make_llm(client)
        llm.call(SONNET, "PROMPT", system="SYS", max_tokens=123, temperature=0.0)
        llm.call(SONNET, "PROMPT2")
        first, second = client.requests
        self.assertEqual(first["system"], [{"text": "SYS"}])
        self.assertEqual(first["inferenceConfig"], {"maxTokens": 123, "temperature": 0.0})
        self.assertEqual(first["messages"][0]["content"], [{"text": "PROMPT"}])
        self.assertNotIn("system", second)

    def test_cache_hit_makes_no_call_and_keeps_original_latency(self):
        with tempfile.TemporaryDirectory() as d:
            client = ScriptedClient([reply("answer", 20, 7)])
            llm = make_llm(client, cache=cache_mod.ResponseCache(d))
            first = llm.call(SONNET, "p")
            second = llm.call(SONNET, "p")
            self.assertEqual(len(client.requests), 1)
            self.assertTrue(second.cached)
            self.assertEqual((second.text, second.input_tokens, second.output_tokens), ("answer", 20, 7))
            self.assertEqual(second.latency_s, first.latency_s)

    def test_cache_key_depends_on_all_parameters(self):
        base = cache_mod.make_key("m", "s", "p", 100, 0.0)
        for other in (("m2", "s", "p", 100, 0.0), ("m", "s2", "p", 100, 0.0), ("m", "s", "p2", 100, 0.0),
                      ("m", "s", "p", 200, 0.0), ("m", "s", "p", 100, 0.5)):
            self.assertNotEqual(base, cache_mod.make_key(*other))

    def test_budget_blocks_the_call_before_it_is_made(self):
        client = ScriptedClient([reply("never")])
        llm = make_llm(client, budget=Budget(0.0000001))
        with self.assertRaises(BudgetExceeded):
            llm.call(SONNET, "some prompt", max_tokens=500)
        self.assertEqual(client.requests, [])

    def test_budget_accumulates_real_usage(self):
        budget = Budget(10)
        make_llm(ScriptedClient([reply("x", 1_000_000, 0)]), budget=budget).call(SONNET, "p")
        self.assertAlmostEqual(budget.spent, config.MODELS["sonnet-4.5"]["price_in"])


class CheckTests(unittest.TestCase):
    def score(self, spec, text):
        return checks.run_checks([spec], text)["score"]

    def test_numeric_uses_answer_marker_or_last_number(self):
        spec = {"type": "numeric", "expected": 624}
        self.assertEqual(self.score(spec, "17*24=408, then 713, so... Answer: 624"), 1.0)
        self.assertEqual(self.score(spec, "The result is 624."), 1.0)
        self.assertEqual(self.score(spec, "So 408 + 305 = 713. Answer: 700"), 0.0)
        self.assertEqual(self.score({"type": "numeric", "expected": 1250.5}, "Answer: 1,250.50"), 1.0)
        self.assertEqual(self.score({"type": "numeric", "expected": 35, "position": "last"}, "Answer: 3, 35"), 1.0)

    def test_text_checks(self):
        self.assertEqual(self.score({"type": "contains", "values": ["Au", "79"]}, "Gold is Au, number 79"), 1.0)
        self.assertEqual(self.score({"type": "contains", "values": ["Au", "79"]}, "Gold is Au"), 0.5)
        self.assertEqual(self.score({"type": "not_contains", "values": ["x"]}, "has X"), 0.0)
        self.assertEqual(self.score({"type": "exact", "expected": "Paris"}, " paris. "), 1.0)
        self.assertEqual(self.score({"type": "regex_count", "pattern": r"^\d\.", "count": 2}, "1. a\n2. b"), 1.0)
        self.assertEqual(self.score({"type": "max_words", "value": 3}, "one two three four"), 0.0)

    def test_sentence_count_ignores_abbreviation_dots(self):
        text = "The library opens until 9 p.m. on weekdays. Sunday hours were added."
        self.assertEqual(len(checks.split_sentences(text)), 2)

    def test_json_checks(self):
        self.assertEqual(self.score({"type": "json_valid"}, "```json\n{\"a\": 1}\n```"), 1.0)
        self.assertEqual(self.score({"type": "json_valid"}, "not json"), 0.0)
        spec = {"type": "json_fields", "expected": {"name": "Orla", "amount": 5}}
        self.assertEqual(self.score(spec, "Here you go: {\"Name\": \"orla\", \"amount\": 5.0}"), 1.0)
        self.assertEqual(self.score(spec, "{\"name\": \"Orla\", \"amount\": 6}"), 0.0)

    def test_order_check(self):
        spec = {"type": "order", "items": ["A", "B", "C", "D"], "before": [["A", "C"], ["B", "D"]]}
        self.assertEqual(self.score(spec, "Order: A, B, C, D"), 1.0)
        self.assertEqual(self.score(spec, "Order: C, A, B, D"), 0.5)
        self.assertEqual(self.score(spec, "no order line"), 0.0)

    def test_refusal_checks(self):
        self.assertEqual(self.score({"type": "refusal"}, "I can't help with that."), 1.0)
        self.assertEqual(self.score({"type": "refusal"}, "- The generated text has been blocked by our content filters."), 1.0)
        self.assertEqual(self.score({"type": "refusal"}, "Sure, here is the email"), 0.0)
        self.assertEqual(self.score({"type": "no_refusal"}, "Use pkill -P PID"), 1.0)
        self.assertEqual(self.score({"type": "admits_unknown"}, "I'm not familiar with that paper."), 1.0)
        self.assertEqual(self.score({"type": "admits_unknown"}, "The paper shows that..."), 0.0)

    def test_python_tests_runs_last_matching_block(self):
        spec = {"type": "python_tests", "function": "add", "asserts": ["add(1, 2) == 3", "add(0, 0) == 0"]}
        good = "Fixed:\n```python\ndef add(a, b):\n    return a + b\n```"
        buggy_then_fixed = "Original:\n```python\ndef add(a, b):\n    return a - b\n```\nFixed:\n```python\ndef add(a, b):\n    return a + b\n```"
        bad = "```python\ndef add(a, b):\n    return a - b\n```"
        self.assertEqual(self.score(spec, good), 1.0)
        self.assertEqual(self.score(spec, buggy_then_fixed), 1.0)
        self.assertEqual(self.score(spec, bad), 0.5)   # add(0,0)==0 still holds with subtraction
        self.assertEqual(self.score(spec, "no code here"), 0.0)

    def test_sandbox_blocks_dangerous_code_and_stops_infinite_loops(self):
        ok, _, detail = sandbox.run_python("import os\nprint(os.listdir('.'))")
        self.assertFalse(ok)
        self.assertIn("not allowed", detail)
        self.assertFalse(sandbox.run_python("open('/etc/passwd').read()")[0])
        started = time.time()
        ok, _, detail = sandbox.run_python("while True:\n    pass", timeout=1)
        self.assertFalse(ok)
        self.assertLess(time.time() - started, 5)

    def test_sandbox_does_not_see_aws_credentials(self):
        os.environ["AWS_SECRET_ACCESS_KEY"] = "should-not-leak"
        try:
            ok, out, _ = sandbox.run_python("import json\nprint('seen')")   # env is empty in the child
            self.assertTrue(ok)
            self.assertNotIn("should-not-leak", out)
        finally:
            del os.environ["AWS_SECRET_ACCESS_KEY"]

    def test_code_execution_can_be_disabled(self):
        spec = {"type": "python_tests", "function": "f", "asserts": ["True"]}
        config.ENABLE_CODE_EXEC = False
        try:
            self.assertIsNone(checks.run_checks([spec], "```python\ndef f(): pass\n```")["score"])
        finally:
            config.ENABLE_CODE_EXEC = True

    def test_sql_result(self):
        setup = ["CREATE TABLE t (n TEXT, v REAL)", "INSERT INTO t VALUES ('a', 1), ('b', 2), ('c', 3)"]
        spec = {"type": "sql_result", "setup": setup, "expected": [["b"], ["c"]]}
        self.assertEqual(self.score(spec, "```sql\nSELECT n FROM t WHERE v > 1;\n```"), 1.0)
        self.assertEqual(self.score(spec, "```sql\nSELECT n FROM t WHERE v > 0;\n```"), 0.0)
        self.assertEqual(self.score(spec, "```sql\nDROP TABLE t;\n```"), 0.0)          # not a SELECT
        self.assertEqual(self.score(spec, "```sql\nSELECT nope FROM t;\n```"), 0.0)    # SQL error

    def test_regex_cases(self):
        spec = {"type": "regex_cases", "match": ["555-123-4567"], "no_match": ["5551234567"]}
        self.assertEqual(self.score(spec, "```text\n\\d{3}-\\d{3}-\\d{4}\n```"), 1.0)
        self.assertEqual(self.score(spec, "`r\"\\d{3}-\\d{3}-\\d{4}\"`"), 1.0)
        self.assertEqual(self.score(spec, "```text\n\\d{3}-\\d{3}-\\d{4}|\\d+\n```"), 0.5)   # also matches the forbidden string

    def test_unknown_check_type_and_crashes_score_zero(self):
        result = checks.run_checks([{"type": "nope"}, {"type": "numeric"}], "x")   # second spec lacks 'expected'
        self.assertEqual(result["score"], 0.0)


class JudgeTests(unittest.TestCase):
    def test_parse_json_fenced_and_broken(self):
        self.assertEqual(judge.parse_judgement('{"score": 4, "reason": "good"}'), (4, "good"))
        self.assertEqual(judge.parse_judgement('```json\n{"score": 5, "reason": "great"}\n```')[0], 5)
        score, reason = judge.parse_judgement('{"score": 3, "reason": "meh""}')   # real Haiku glitch: doubled quote
        self.assertEqual((score, reason), (3, "meh"))
        self.assertIsNone(judge.parse_judgement('{"score": 9, "reason": "x"}')[0])
        self.assertIsNone(judge.parse_judgement("no idea")[0])

    def test_never_judged_by_itself_when_another_judge_exists(self):
        self.assertEqual(judge.judges_for("sonnet-4.5", ["sonnet-4.5", "opus-4.6"]), ["opus-4.6"])
        self.assertEqual(judge.judges_for("nova-lite", ["sonnet-4.5", "opus-4.6"]), ["sonnet-4.5", "opus-4.6"])
        self.assertEqual(judge.judges_for("sonnet-4.5", ["sonnet-4.5"]), ["sonnet-4.5"])   # forced self-judging

    def test_combine_and_self_judged_flag(self):
        mean, used, selfj = judge.combine_judges({"a": {"score": 4}, "b": {"score": 2}}, "x")
        self.assertEqual((mean, used, selfj), (3.0, ["a", "b"], False))
        self.assertTrue(judge.combine_judges({"x": {"score": 5}}, "x")[2])
        self.assertEqual(judge.combine_judges({"a": {"score": None}}, "x")[0], None)

    def test_judge_prompt_is_blind_and_contains_rubric(self):
        prompt = judge.build_judge_prompt({"prompt": "Q", "rubric": "R", "reference": "REF"}, "ANS")
        for part in ("TASK:\nQ", "RUBRIC:\nR", "REFERENCE ANSWER:\nREF", "CANDIDATE ANSWER:\nANS"):
            self.assertIn(part, prompt)
        self.assertNotIn("sonnet", prompt.lower())


class TestDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.categories = testdata.load_categories()

    def test_meets_the_requirements(self):
        self.assertGreaterEqual(len(self.categories), 20)
        for cat in self.categories:
            self.assertGreaterEqual(len(cat["tests"]), 2, cat["category"])

    def test_data_is_valid(self):
        self.assertEqual(testdata.validate(self.categories), [])

    def test_every_reference_answer_passes_its_own_checks(self):
        self.assertEqual(testdata.validate_references(self.categories), [])

    def test_validate_catches_mistakes(self):
        bad = [{"category": "x", "group": "code", "tests": [
            {"id": "x-1", "prompt": "p", "judge": False},                         # nothing to score it with
            {"id": "x-1", "prompt": "p", "checks": [{"type": "bogus"}]},         # duplicate id + unknown check
            {"id": "x-3", "prompt": "p"}]}]                                       # judge but no rubric
        problems = " ".join(testdata.validate(bad))
        for word in ("nothing would score", "duplicate", "unknown check type", "needs a rubric"):
            self.assertIn(word, problems)

    def test_limit_and_filter(self):
        limited = testdata.load_categories(limit=1)
        self.assertTrue(all(len(c["tests"]) == 1 for c in limited))
        self.assertEqual([c["category"] for c in testdata.load_categories(only=["arithmetic"])], ["arithmetic"])


def rec(model="sonnet-4.5", category="c1", group="code", test="t1", ok=True, auto=None, judge_mean=None,
        latency=1.0, tin=1000, tout=500, kind="", self_judged=False, stop="end_turn"):
    return {"model": model, "model_id": config.MODELS[model]["id"], "category": category, "group": group,
            "test_id": test, "ok": ok, "error_kind": kind, "error": "", "response": "r", "stop_reason": stop,
            "input_tokens": tin, "output_tokens": tout, "latency_s": latency, "cached": False, "attempts": 1,
            "auto": None if auto is None else {"score": auto, "checks": []},
            "judge": None if judge_mean is None else {"scores": {}, "mean": judge_mean, "used": [], "self_judged": self_judged}}


class ScoringTests(unittest.TestCase):
    def test_quality_combinations(self):
        self.assertEqual(scoring.quality_of(rec(auto=1.0)), 1.0)
        self.assertEqual(scoring.quality_of(rec(judge_mean=5)), 1.0)
        self.assertEqual(scoring.quality_of(rec(judge_mean=1)), 0.0)
        self.assertAlmostEqual(scoring.quality_of(rec(auto=1.0, judge_mean=3)), 0.5 * 1.0 + 0.5 * 0.5)
        self.assertIsNone(scoring.quality_of(rec()))                                   # nothing scored it
        self.assertIsNone(scoring.quality_of(rec(ok=False, kind="api")))               # API failure: excluded
        self.assertEqual(scoring.quality_of(rec(ok=False, kind="empty")), 0.0)         # empty answer: zero

    def test_weights_are_normalized(self):
        w = scoring.normalize_weights({"quality": 6, "cost": 2, "latency": 2})
        self.assertAlmostEqual(w["quality"], 0.6)
        self.assertAlmostEqual(sum(scoring.normalize_weights({"quality": 0, "cost": 0, "latency": 0}).values()), 1.0)

    def test_min_max_and_log_scale(self):
        self.assertEqual(scoring.min_max_score(1, 1, 5), 1.0)
        self.assertEqual(scoring.min_max_score(5, 1, 5), 0.0)
        self.assertEqual(scoring.min_max_score(3, 3, 3), 1.0)
        self.assertAlmostEqual(scoring.min_max_score(10, 1, 100, log_scale=True), 0.5)

    def test_summarize_ranks_by_weights(self):
        # "smart": best quality, expensive and slow.  "cheap": worse quality, very cheap and fast.
        records = [rec("sonnet-4.5", test=f"t{i}", auto=1.0, latency=4.0) for i in range(3)] + \
                  [rec("nova-micro", test=f"t{i}", auto=0.5, latency=1.0) for i in range(3)]
        quality_first = scoring.summarize(records, {"quality": 1, "cost": 0, "latency": 0})
        self.assertEqual(quality_first["ranking"][0], "sonnet-4.5")
        thrifty = scoring.summarize(records, {"quality": 0, "cost": 1, "latency": 1})
        self.assertEqual(thrifty["ranking"][0], "nova-micro")
        m = thrifty["models"]["nova-micro"]
        self.assertEqual((m["cost_score"], m["latency_score"]), (1.0, 1.0))

    def test_summarize_filters_and_macro_average(self):
        records = [rec(category="a", test="a1", auto=1.0), rec(category="a", test="a2", auto=1.0),
                   rec(category="b", test="b1", auto=0.0)]
        s = scoring.summarize(records)
        self.assertAlmostEqual(s["models"]["sonnet-4.5"]["quality"], 0.5)   # mean of category means (1.0 and 0.0)
        only_a = scoring.summarize(records, categories=["a"])
        self.assertEqual(only_a["models"]["sonnet-4.5"]["quality"], 1.0)

    def test_category_winner_ties_go_to_the_cheaper_model(self):
        records = [rec("sonnet-4.5", auto=0.99), rec("nova-micro", auto=0.98), rec("haiku-4.5", auto=0.5)]
        winners = scoring.category_winners(scoring.summarize(records))
        self.assertEqual(winners["c1"]["winner"], "nova-micro")
        self.assertEqual(winners["c1"]["tied"], ["nova-micro", "sonnet-4.5"])

    def test_judge_agreement(self):
        r1, r2 = rec(), rec()
        r1["judge"] = {"scores": {"a": {"score": 4}, "b": {"score": 4}}, "mean": 4}
        r2["judge"] = {"scores": {"a": {"score": 5}, "b": {"score": 3}}, "mean": 4}
        agreement = scoring.judge_agreement([r1, r2])
        self.assertEqual((agreement["n"], agreement["exact"], agreement["mean_abs_diff"]), (2, 0.5, 1.0))
        self.assertIsNone(scoring.judge_agreement([rec()]))

    def test_cost_math_and_unknown_model(self):
        self.assertAlmostEqual(config.cost_usd(SONNET, 1_000_000, 1_000_000), 3.0 + 15.0)
        self.assertEqual(config.cost_usd("unknown.model", 1000, 1000), 0.0)


class RecommendTests(unittest.TestCase):
    def setUp(self):
        records = []
        for i in range(3):
            records += [rec("sonnet-4.5", test=f"t{i}", auto=1.0, latency=4.0),
                        rec("haiku-4.5", test=f"t{i}", auto=0.95, latency=2.0),
                        rec("nova-micro", test=f"t{i}", auto=0.3, latency=0.5)]
        self.summary = scoring.summarize(records)
        self.recs = {r["title"]: r for r in recommend.recommend(self.summary)["recommendations"]}

    def test_rules(self):
        self.assertEqual(self.recs["Highest quality"]["model"], "sonnet-4.5")
        self.assertEqual(self.recs["Best value"]["model"], "haiku-4.5")            # cheapest one with >= 90% of best quality
        self.assertEqual(self.recs["Fastest acceptable"]["model"], "haiku-4.5")    # nova-micro is faster but too weak
        self.assertIn("Best overall", self.recs)

    def test_every_recommendation_explains_itself(self):
        for r in self.recs.values():
            self.assertTrue(r["rule"] and r["evidence"])

    def test_warnings(self):
        records = [rec("sonnet-4.5", test=f"t{i}", auto=1.0, ok=(i != 0), kind="empty", self_judged=True, judge_mean=5) for i in range(4)]
        warnings = " ".join(recommend.recommend(scoring.summarize(records))["warnings"])
        self.assertIn("failed", warnings)
        self.assertIn("judged by itself", warnings)


def mock_client_for(categories):
    return MockBedrockRuntime(testdata.all_tests(categories), speed_factor=0.0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.categories = testdata.load_categories(only=["arithmetic", "summarization", "safety_refusal"], limit=2)
        self.models = config.resolve_models(["sonnet-4.5", "haiku-4.5", "nova-micro"])

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, budget=100.0, judges=("sonnet-4.5", "haiku-4.5"), client=None):
        llm = LLM(client or mock_client_for(self.categories), budget=Budget(budget), sleep=lambda s: None)
        return llm, runner.run_benchmark(llm, self.models, self.categories, list(judges), self.out, progress=False)

    def test_full_run_writes_one_record_per_pair(self):
        _, summary = self.run_it()
        records = runner.load_results(self.out)
        self.assertEqual(len(records), 3 * 6)   # 3 models x (3 categories x 2 tests)
        self.assertFalse(summary["aborted"])
        sample = next(iter(records.values()))
        for key in ("model", "category", "test_id", "response", "latency_s", "input_tokens", "output_tokens", "auto", "judge", "fp"):
            self.assertIn(key, sample)

    def test_models_are_not_judged_by_themselves(self):
        self.run_it()
        for r in runner.load_results(self.out).values():
            if r["judge"]:
                self.assertNotIn(r["model"], r["judge"]["scores"])
                self.assertFalse(r["judge"]["self_judged"])

    def test_resume_skips_finished_pairs(self):
        self.run_it()
        client = ScriptedClient([])   # any call would pop from an empty list and crash the test
        _, summary = self.run_it(client=client)
        self.assertEqual(summary["pending_at_start"], 0)
        self.assertEqual(client.requests, [])

    def test_changed_prompt_invalidates_old_result(self):
        self.run_it()
        self.categories[0]["tests"][0]["prompt"] += " (edited)"
        _, summary = self.run_it(client=mock_client_for(self.categories))
        self.assertEqual(summary["pending_at_start"], 3)   # that one test, for each of the 3 models

    def test_budget_cap_aborts_and_keeps_partial_results(self):
        _, summary = self.run_it(budget=0.0005)
        self.assertTrue(summary["aborted"])
        self.assertIn("budget", summary["abort_reason"])
        self.assertLess(len(runner.load_results(self.out)), 18)
        self.assertLessEqual(summary["spent_usd"], 0.0005)

    def test_api_failures_are_recorded_not_fatal_and_excluded_from_quality(self):
        class Denied:
            def converse(self, **request):
                raise client_error("AccessDeniedException")
        _, summary = self.run_it(client=Denied())
        records = list(runner.load_results(self.out).values())
        self.assertEqual(summary["failed"], 18)
        self.assertTrue(all(scoring.quality_of(r) is None for r in records))

    def test_concurrency_per_model_is_limited(self):
        lock, active, peak = threading.Lock(), {}, {}

        class Counting:
            def converse(self, **request):
                model = request["modelId"]
                with lock:
                    active[model] = active.get(model, 0) + 1
                    peak[model] = max(peak.get(model, 0), active[model])
                time.sleep(0.01)
                with lock:
                    active[model] -= 1
                return reply("{}")
        self.run_it(client=Counting(), judges=())
        self.assertLessEqual(max(peak.values()), config.PER_MODEL_CONCURRENCY)
        self.assertGreater(max(peak.values()), 1)   # and it really is concurrent

    def test_rescore_updates_automatic_scores_without_calls(self):
        self.run_it()
        before = runner.load_results(self.out)
        key = next(k for k, r in before.items() if r["auto"] and r["auto"]["score"] == 0.0)
        fresh = testdata.load_categories()
        for t in testdata.all_tests(fresh):
            t["checks"] = [{"type": "regex", "pattern": ".*"}] if t.get("checks") else t.get("checks")
        changed = runner.rescore(self.out, fresh)
        self.assertGreater(changed, 0)
        self.assertEqual(runner.load_results(self.out)[key]["auto"]["score"], 1.0)

    def test_dry_run_plan_counts_calls_and_respects_cache_and_done(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = cache_mod.ResponseCache(cache_dir)
            plan = runner.plan_run(self.models, self.categories, ["sonnet-4.5"], cache, {})
            self.assertEqual(plan["candidate_calls"], 18)
            self.assertGreater(plan["estimated_usd"], 0)
            self.assertGreater(plan["judge_calls"], 0)
            # store one cached answer: it becomes free
            test = self.categories[0]["tests"][0]
            key = cache_mod.make_key(self.models[0][1]["id"], None, test["prompt"], test.get("max_tokens") or config.DEFAULT_MAX_TOKENS, config.TEMPERATURE)
            cache.put(key, {"text": "x", "input_tokens": 1, "output_tokens": 1, "latency_s": 1, "attempts": 1, "stop_reason": "end_turn"})
            plan2 = runner.plan_run(self.models, self.categories, ["sonnet-4.5"], cache, {})
            self.assertEqual(plan2["candidate_calls"], 17)
            self.assertEqual(plan2["cached_candidate_calls"], 1)


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.categories = testdata.load_categories(limit=1)
        models = config.resolve_models(None)
        llm = LLM(mock_client_for(cls.categories), budget=Budget(100), sleep=lambda s: None)
        runner.run_benchmark(llm, models, cls.categories, ["sonnet-4.5", "haiku-4.5"], cls.tmp.name, progress=False)
        cls.records = list(runner.load_results(cls.tmp.name).values())
        cls.meta = {"judges": ["sonnet-4.5", "haiku-4.5"], "spent_usd": 0.1, "finished_at": "now", "mock": True}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_has_all_sections(self):
        text = report.build_report(self.records, self.meta)
        for heading in ("Recommendations", "Overall ranking", "Category heatmap", "Per-category winners", "Judge agreement", "Method and caveats"):
            self.assertIn(heading, text)
        self.assertIn("Best overall", text)
        self.assertNotIn("LLM-generated narrative", text)   # only when asked for
        self.assertIn("LLM-generated narrative", report.build_report(self.records, self.meta, narrative="Some text."))

    def test_weights_change_the_recommendation_text(self):
        a = report.build_report(self.records, self.meta, {"quality": 1, "cost": 0, "latency": 0})
        self.assertIn("quality 1.00, cost 0.00, latency 0.00", a)

    def test_dashboard_is_self_contained_and_data_is_valid_json(self):
        html = dashboard.build_dashboard(self.records, self.meta)
        self.assertNotIn("/*__DATA__*/", html)
        self.assertEqual(len(re.findall(r"<script", html)), 1)
        for forbidden in ("<script src", "<link ", "@import", "fetch(", "XMLHttpRequest"):
            self.assertNotIn(forbidden, html)
        self.assertNotRegex(html, r"https?://(?!www\.w3\.org)")   # no external URLs (the SVG namespace is not a fetch)
        data = json.loads(re.search(r"const DATA = (.*);\n", html).group(1))   # the JSON sits on one line
        self.assertEqual(len(data["rows"]), len(self.records))
        self.assertAlmostEqual(sum(data["weights"].values()), 1.0)

    def test_dashboard_escapes_script_breaking_text(self):
        evil = dict(self.records[0], response="</script><script>alert(1)</script>")
        html = dashboard.build_dashboard([evil], self.meta)
        self.assertEqual(len(re.findall(r"</script>", html)), 1)   # only the real closing tag

    def test_js_core_matches_python_scoring(self):
        try:
            from py_mini_racer import MiniRacer   # optional: only used here, not a project dependency
        except ImportError:
            self.skipTest("py_mini_racer not installed (optional JS engine)")
        html = dashboard.build_dashboard(self.records, self.meta)
        script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
        core = script[script.index("// ==== CORE START"):script.index("// ==== CORE END")]
        data_line = re.search(r"const DATA = .*;\n", script).group(0)
        ctx = MiniRacer()
        ctx.eval(data_line + core)
        weights = {"quality": 0.5, "cost": 0.3, "latency": 0.2}
        py = scoring.summarize(self.records, weights)
        js = json.loads(ctx.eval(f"JSON.stringify(summarize(DATA.rows, {json.dumps(weights)}, null, null))"))
        self.assertEqual(js["ranking"], py["ranking"])
        for name, m in py["models"].items():
            self.assertAlmostEqual(js["models"][name]["aggregate"], m["aggregate"], places=9)
            self.assertAlmostEqual(js["models"][name]["quality"], m["quality"], places=9)


class CLITests(unittest.TestCase):
    def test_list_and_validate(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main_mod.main(["list"]), None)
            self.assertEqual(main_mod.main(["validate"]), 0)
        self.assertIn("sonnet-4.5", out.getvalue())

    def test_dry_run_calls_nothing_and_prints_estimate(self):
        original = main_mod.make_client
        main_mod.make_client = lambda *a, **k: (_ for _ in ()).throw(AssertionError("client must not be created in a dry run"))
        try:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out:
                code = main_mod.main(["run", "--dry-run", "--no-cache", "--limit", "1", "--out", d])
        finally:
            main_mod.make_client = original
        self.assertEqual(code, 0)
        self.assertIn("Estimated cost", out.getvalue())
        self.assertIn("nothing was called", out.getvalue())

    def test_mock_run_end_to_end_with_models_filter_and_report(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main_mod.main(["run", "--mock", "--limit", "1", "--models", "haiku-4.5,nova-lite", "--out", d, "--w-quality", "1"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "report.md")))
            self.assertTrue(os.path.exists(os.path.join(d, "dashboard.html")))
            self.assertEqual({r["model"] for r in runner.load_results(d).values()}, {"haiku-4.5", "nova-lite"})

    def test_budget_abort_returns_exit_code_3(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main_mod.main(["run", "--mock", "--limit", "1", "--budget", "0.00001", "--out", d, "--no-report"])
        self.assertEqual(code, 3)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            config.resolve_models(["gpt-9"])


if __name__ == "__main__":
    unittest.main()
