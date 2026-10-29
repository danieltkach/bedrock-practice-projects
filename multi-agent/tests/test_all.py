"""Offline tests: no AWS, no network. Run with: ./run.sh test   (or python -m unittest discover)"""
import json
import threading
import time
import unittest

from botocore.exceptions import ClientError

import config
from agents import build_agent_prompt, AgentResult
from coordinator import Plan, Subtask, parse_plan
from llm import LLM, LLMError, extract_text
from metrics import Tracker, cost_usd
from mock_client import MockBedrockRuntime
from orchestrator import answer_query, split_into_waves


def throttle():
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse")


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


def make_llm(client):
    return LLM(client, Tracker(), sleep=lambda s: None)   # no real waiting


class ParsingTests(unittest.TestCase):
    def test_extract_text_skips_non_text_blocks(self):
        resp = {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": "hi"}]}}}
        self.assertEqual(extract_text(resp), "hi")

    def test_reasoning_only_response_raises(self):
        client = ScriptedClient([{"output": {"message": {"content": [{"reasoningContent": {}}]}},
                                  "stopReason": "max_tokens", "usage": {}}])
        with self.assertRaises(LLMError):
            make_llm(client).call("direct", "s", "p")


class RetryTests(unittest.TestCase):
    def test_retries_throttling_then_succeeds(self):
        client = ScriptedClient([throttle(), throttle(), reply("ok")])
        llm = make_llm(client)
        self.assertEqual(llm.call("direct", "s", "p"), "ok")
        self.assertEqual(llm.tracker.records[0].attempts, 3)

    def test_gives_up_after_max_attempts(self):
        client = ScriptedClient([throttle()] * config.MAX_ATTEMPTS)
        llm = make_llm(client)
        with self.assertRaises(LLMError):
            llm.call("direct", "s", "p")
        self.assertFalse(llm.tracker.records[0].ok)

    def test_non_retryable_error_fails_immediately(self):
        err = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse")
        client = ScriptedClient([err])
        with self.assertRaises(LLMError):
            make_llm(client).call("direct", "s", "p")
        self.assertEqual(len(client.requests), 1)

    def test_request_shape(self):
        client = ScriptedClient([reply("ok")])
        make_llm(client).call("analysis", "SYS", "PROMPT")
        req = client.requests[0]
        self.assertEqual(req["modelId"], config.ROLE_MODEL["analysis"])
        self.assertEqual(req["system"], [{"text": "SYS"}])
        self.assertEqual(req["messages"][0]["content"], [{"text": "PROMPT"}])


class PlanTests(unittest.TestCase):
    def plan_json(self, subtasks, **extra):
        return json.dumps({"reasoning": "r", "direct_answer": "", "subtasks": subtasks, **extra})

    def test_valid_multi_plan_with_code_fence(self):
        text = "```json\n" + self.plan_json([
            {"id": "s1", "agent": "analysis", "task": "a", "depends_on": []},
            {"id": "s2", "agent": "writing", "task": "w", "depends_on": ["s1"]}]) + "\n```"
        plan = parse_plan(text, "q")
        self.assertEqual(plan.route, "multi")
        self.assertFalse(plan.fallback)

    def test_direct_and_single_routes(self):
        self.assertEqual(parse_plan(self.plan_json([], direct_answer="hi"), "q").route, "direct")
        one = [{"id": "s1", "agent": "code", "task": "t", "depends_on": []}]
        self.assertEqual(parse_plan(self.plan_json(one), "q").route, "single")

    def test_bad_inputs_fall_back(self):
        bad = [
            "not json at all",
            self.plan_json([{"id": "s1", "agent": "poet", "task": "t"}]),                       # unknown agent
            self.plan_json([{"id": "s1", "agent": "code", "task": "t", "depends_on": ["zz"]}]),  # missing dep
            self.plan_json([{"id": "s1", "agent": "code", "task": "t", "depends_on": ["s2"]},
                            {"id": "s2", "agent": "code", "task": "t", "depends_on": ["s1"]}]),  # cycle
            self.plan_json([{"id": "s1", "agent": "code", "task": "t"}] * 2),                    # duplicate ids
            self.plan_json([{"id": f"s{i}", "agent": "code", "task": "t"}
                            for i in range(config.MAX_SUBTASKS + 1)]),                            # too many
        ]
        for text in bad:
            plan = parse_plan(text, "my question")
            self.assertTrue(plan.fallback, text[:40])
            self.assertEqual(plan.subtasks[0].task, "my question")

    def test_waves(self):
        subs = [Subtask("s1", "analysis", "a"), Subtask("s2", "code", "c"),
                Subtask("s3", "writing", "w", ["s1", "s2"])]
        waves = split_into_waves(subs)
        self.assertEqual([[s.id for s in w] for w in waves], [["s1", "s2"], ["s3"]])


class MetricsTests(unittest.TestCase):
    def test_cost_math(self):
        # 1M input + 1M output on the Sonnet price entry
        self.assertAlmostEqual(cost_usd(config.SONNET, 1_000_000, 1_000_000),
                               sum(config.PRICES_PER_MILLION[config.SONNET]))

    def test_unknown_model_costs_zero(self):
        self.assertEqual(cost_usd("some.unknown-model", 1000, 1000), 0.0)


class EndToEndMockTests(unittest.TestCase):
    def run_query(self, query, delay=0.0):
        llm = LLM(MockBedrockRuntime(delay=delay), Tracker(), sleep=lambda s: None)
        return answer_query(llm, query), llm

    def test_direct_uses_only_coordinator(self):
        result, llm = self.run_query("Hello there")
        self.assertEqual(result.plan.route, "direct")
        self.assertEqual([r.role for r in llm.tracker.records], ["coordinator"])

    def test_single_skips_aggregator(self):
        result, llm = self.run_query("Write a python function to reverse a string")
        self.assertEqual(result.plan.route, "single")
        roles = [r.role for r in llm.tracker.records]
        self.assertNotIn("aggregator", roles)
        self.assertIn("(mock code)", result.answer)

    def test_multi_aggregates(self):
        result, llm = self.run_query("Compare SQS and Kinesis, write python code, and draft an email")
        self.assertEqual(result.plan.route, "multi")
        roles = [r.role for r in llm.tracker.records]
        self.assertEqual(roles[0], "coordinator")
        self.assertEqual(roles[-1], "aggregator")
        self.assertIn("Final answer", result.answer)

    def test_independent_subtasks_run_in_parallel(self):
        # analysis + code are independent: with 0.3s per call they should overlap.
        started = time.perf_counter()
        self.run_query("Analyze trade-offs and write python code", delay=0.3)
        elapsed = time.perf_counter() - started
        # sequential would be coordinator + analysis + code + aggregator = 4 x 0.3 = 1.2s
        self.assertLess(elapsed, 1.0)

    def test_dependency_output_reaches_next_agent(self):
        res = AgentResult("s1", "analysis", True, "FINDINGS-XYZ")
        prompt = build_agent_prompt("summarize", "orig", [res])
        self.assertIn("FINDINGS-XYZ", prompt)

    def test_failed_agent_does_not_kill_the_answer(self):
        plan_text = json.dumps({"reasoning": "r", "direct_answer": "", "subtasks": [
            {"id": "s1", "agent": "analysis", "task": "a", "depends_on": []},
            {"id": "s2", "agent": "code", "task": "c", "depends_on": []}]})
        denied = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "Converse")
        # Parallel order of the two agent calls is not fixed, so use a thread-safe client by role.
        lock = threading.Lock()

        class RoleClient:
            def converse(self, **request):
                system = request["system"][0]["text"]
                with lock:
                    if "ROUTING COORDINATOR" in system:
                        return reply(plan_text)
                    if "CODE AGENT" in system:
                        raise denied
                    if "AGGREGATOR" in system:
                        prompt = request["messages"][0]["content"][0]["text"]
                        assert "UNAVAILABLE" in prompt
                        return reply("merged with a gap")
                    return reply("analysis ok")

        result = answer_query(make_llm(RoleClient()), "q")
        self.assertEqual(result.answer, "merged with a gap")
        self.assertFalse(result.results["s2"].ok)

    def test_bad_coordinator_output_falls_back_to_analysis(self):
        client = ScriptedClient([reply("I refuse to output JSON"), reply("fallback answer")])
        result = answer_query(make_llm(client), "q")
        self.assertTrue(result.plan.fallback)
        self.assertEqual(result.answer, "fallback answer")


if __name__ == "__main__":
    unittest.main()
