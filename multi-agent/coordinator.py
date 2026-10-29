"""The coordinator: asks a cheap model for a PLAN, then validates it in plain Python.

The model is good at understanding the question; Python is good at enforcing rules. So we never
trust the model's JSON blindly: parse_plan() checks it and falls back to a safe plan if needed.
"""
import json
import logging
import re
from dataclasses import dataclass, field

import config
from agents import AGENT_DESCRIPTIONS
from llm import LLMError

log = logging.getLogger("multi-agent")


@dataclass
class Subtask:
    id: str
    agent: str
    task: str
    depends_on: list = field(default_factory=list)


@dataclass
class Plan:
    subtasks: list            # list[Subtask]; empty means "answer directly"
    direct_answer: str = ""   # used only when subtasks is empty
    reasoning: str = ""
    fallback: bool = False    # True when the model's plan was unusable

    @property
    def route(self):
        """Derived from the subtasks, so the model cannot give an inconsistent route."""
        if not self.subtasks:
            return "direct"
        return "single" if len(self.subtasks) == 1 else "multi"


def planner_system_prompt():
    agents = "\n".join(f'- "{name}": {desc}' for name, desc in AGENT_DESCRIPTIONS.items())
    return (
        "You are the ROUTING COORDINATOR of a multi-agent system. Read the user's question and "
        "reply with ONLY a JSON object, no other text.\n\n"
        f"Available specialist agents:\n{agents}\n\n"
        "Rules:\n"
        "1. Trivial question (greeting, one-line fact, small talk): use an empty \"subtasks\" list "
        "and put a short answer in \"direct_answer\".\n"
        "2. Needs one kind of skill: ONE subtask.\n"
        "3. Needs several skills: several subtasks. Subtasks that do not need each other's output "
        "must have an empty \"depends_on\" so they run in parallel. Use \"depends_on\" only when a "
        "step really needs another step's result (e.g. writing a summary of an analysis).\n"
        f"4. At most {config.MAX_SUBTASKS} subtasks. Prefer the fewest that do the job: every "
        "subtask costs money.\n"
        "5. Each \"task\" must be a self-contained instruction.\n\n"
        "JSON format:\n"
        '{"reasoning": "one sentence", "direct_answer": "", "subtasks": '
        '[{"id": "s1", "agent": "analysis", "task": "...", "depends_on": []}]}'
    )


def _extract_json(text):
    """Models sometimes wrap JSON in ```json fences or add a sentence. Grab the outer {...}."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(cleaned[start:end + 1])


def _has_cycle(subtasks):
    """Depth-first search for a dependency loop (s1 needs s2, s2 needs s1)."""
    deps = {s.id: s.depends_on for s in subtasks}
    state = {}   # id -> 1 while visiting, 2 when finished

    def visit(node):
        if state.get(node) == 2:
            return False
        if state.get(node) == 1:
            return True
        state[node] = 1
        if any(visit(d) for d in deps[node]):
            return True
        state[node] = 2
        return False

    return any(visit(s.id) for s in subtasks)


def parse_plan(text, query):
    """Turn the model's reply into a validated Plan. Any problem -> a safe fallback plan."""
    try:
        data = _extract_json(text)
        raw = data.get("subtasks") or []
        if not isinstance(raw, list):
            raise ValueError("subtasks is not a list")
        subtasks = []
        for item in raw:
            agent = item.get("agent")
            if agent not in AGENT_DESCRIPTIONS:
                raise ValueError(f"unknown agent {agent!r}")
            task = str(item.get("task", "")).strip()
            if not task:
                raise ValueError("empty task")
            subtasks.append(Subtask(str(item.get("id")), agent, task,
                                    [str(d) for d in item.get("depends_on") or []]))
        if len(subtasks) > config.MAX_SUBTASKS:
            raise ValueError(f"too many subtasks ({len(subtasks)})")
        ids = [s.id for s in subtasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate subtask ids")
        for s in subtasks:
            if any(d not in ids or d == s.id for d in s.depends_on):
                raise ValueError(f"bad dependency in {s.id}")
        if _has_cycle(subtasks):
            raise ValueError("dependency cycle")
        return Plan(subtasks, str(data.get("direct_answer", "")).strip(),
                    str(data.get("reasoning", "")).strip())
    except (ValueError, AttributeError, TypeError) as err:
        log.warning("plan rejected (%s); using fallback", err)
        # Fallback: one analysis agent answers the whole question. Slower but always safe.
        return Plan([Subtask("s1", "analysis", query)], reasoning=f"fallback: {err}", fallback=True)


def make_plan(llm, query, history_text=""):
    """Ask the coordinator model for a plan and validate it."""
    prompt = f"User question:\n{query}"
    if history_text:
        prompt = f"Recent conversation (for follow-up questions):\n{history_text}\n\n{prompt}"
    try:
        reply = llm.call("coordinator", planner_system_prompt(), prompt, temperature=0.0)
    except LLMError as err:
        log.warning("coordinator call failed (%s); using fallback", err)
        return Plan([Subtask("s1", "analysis", query)], reasoning=f"fallback: {err}", fallback=True)
    plan = parse_plan(reply, query)
    log.info("routing route=%s fallback=%s agents=%s reasoning=%s", plan.route, plan.fallback,
             [(s.id, s.agent, s.depends_on) for s in plan.subtasks], plan.reasoning)
    return plan
