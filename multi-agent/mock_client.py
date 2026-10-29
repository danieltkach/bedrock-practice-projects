"""A fake 'bedrock-runtime' client that needs no AWS. Used by --mock mode and by the tests.

It looks at the system prompt to know which role is calling, and returns plausible canned text
with realistic-looking token counts. Its planner picks agents with simple keyword rules, which is
enough to demo every route (direct, single, multi) offline.
"""
import json
import time

ANALYSIS_WORDS = ("analy", "compare", "trade-off", "tradeoff", "pros and cons", "evaluate", "why", "recommend")
WRITING_WORDS = ("write an email", "email", "summar", "draft", "announce", "report", "blog")
CODE_WORDS = ("code", "python", "function", "script", "sql", "implement", "bug")


def _tokens(text):
    return max(1, len(text) // 4)   # rough rule of thumb: 1 token is about 4 characters


def _mock_plan(query):
    q = query.lower()
    wants = [name for name, words in (("analysis", ANALYSIS_WORDS), ("code", CODE_WORDS),
                                      ("writing", WRITING_WORDS)) if any(w in q for w in words)]
    if not wants:
        return {"reasoning": "simple question", "direct_answer": "Hello! (mock answer) Ask me something bigger.",
                "subtasks": []}
    subtasks = []
    for i, agent in enumerate(wants, start=1):
        deps = [f"s{j}" for j in range(1, i)] if agent == "writing" and i > 1 else []
        subtasks.append({"id": f"s{i}", "agent": agent, "task": f"Handle the {agent} part of: {query}",
                         "depends_on": deps})
    return {"reasoning": "mock keyword routing", "direct_answer": "", "subtasks": subtasks}


class MockBedrockRuntime:
    def __init__(self, delay=0.15):
        self.delay = delay   # fake network time, so parallel vs sequential is visible in the report

    def converse(self, **request):
        time.sleep(self.delay)
        system = request["system"][0]["text"]
        prompt = request["messages"][0]["content"][0]["text"]
        if "ROUTING COORDINATOR" in system:
            query = prompt.split("User question:\n")[-1]
            text = json.dumps(_mock_plan(query))
        elif "AGGREGATOR" in system:
            text = "(mock) Final answer: merged the specialist results into one coherent response."
        elif "ANALYSIS AGENT" in system:
            text = "(mock analysis) Option A is cheaper; option B scales better. Recommendation: A first."
        elif "WRITING AGENT" in system:
            text = "(mock writing) Hi team, here is a clear short summary of the findings."
        elif "CODE AGENT" in system:
            text = "(mock code)\n```python\ndef add(a, b):\n    return a + b\n```"
        else:
            text = "(mock) Short direct answer."
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": _tokens(system + prompt), "outputTokens": _tokens(text) + 40},
        }
