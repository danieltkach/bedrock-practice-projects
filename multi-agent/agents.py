"""The specialized agents. An 'agent' here is: a system prompt + a model tier + a role name.

They are deliberately simple. What makes them 'specialized' is the system prompt (who they are,
what they must and must not do) and the model chosen for them in config.ROLE_MODEL.
"""
from dataclasses import dataclass

from llm import LLMError

AGENT_PROMPTS = {
    "analysis": (
        "You are the ANALYSIS AGENT. You reason carefully about a question: compare options, weigh "
        "trade-offs, check assumptions and give a clear conclusion. Be structured (short headings or "
        "bullets) and concrete. State uncertainty honestly. Do not write code unless asked and do not "
        "write long polished prose: other agents handle that."
    ),
    "writing": (
        "You are the WRITING AGENT. You turn material into clear, well-organized text for a human "
        "reader: summaries, emails, announcements, explanations. Match the requested tone and length. "
        "Use only the facts you are given; do not invent numbers or names."
    ),
    "code": (
        "You are the CODE AGENT. You write correct, readable code with brief comments explaining the "
        "why. Put code in fenced blocks with the language name. Mention assumptions and how to run it "
        "in two or three lines at most."
    ),
}

DIRECT_PROMPT = (
    "You are a helpful assistant. Answer the user briefly and accurately in a few sentences."
)

AGGREGATOR_PROMPT = (
    "You are the AGGREGATOR. Several specialist agents each answered part of the user's question. "
    "Merge their results into ONE coherent final answer: remove repetition, resolve contradictions "
    "(say so if you cannot), keep the order logical, keep code blocks intact and unchanged in "
    "meaning, and do not mention the agents or the merging process. If a part is marked "
    "UNAVAILABLE, tell the user briefly which part is missing. Do not add new facts."
)

AGENT_DESCRIPTIONS = {
    "analysis": "reasoning, comparison, trade-offs, evaluation, recommendations, explaining why",
    "writing": "drafting emails, summaries, reports, announcements, polished explanations",
    "code": "writing, fixing or explaining code, SQL, scripts, configurations",
}


@dataclass
class AgentResult:
    subtask_id: str
    agent: str
    ok: bool
    text: str        # the answer, or the error message when ok is False


def build_agent_prompt(task, original_query, dependency_results):
    """The user message sent to a specialist: its task, the overall goal, and earlier results."""
    parts = [f"Overall user request (for context only):\n{original_query}"]
    if dependency_results:
        parts.append("Results from earlier steps you can build on:")
        for res in dependency_results:
            body = res.text if res.ok else f"UNAVAILABLE ({res.text})"
            parts.append(f"--- {res.agent} result ({res.subtask_id}) ---\n{body}")
    parts.append(f"YOUR TASK:\n{task}")
    return "\n\n".join(parts)


def run_agent(llm, subtask, original_query, dependency_results):
    """Run one subtask. Never raises: a failure becomes an AgentResult(ok=False) so the
    other agents can still finish and the aggregator can report the gap."""
    prompt = build_agent_prompt(subtask.task, original_query, dependency_results)
    try:
        text = llm.call(subtask.agent, AGENT_PROMPTS[subtask.agent], prompt)
        return AgentResult(subtask.id, subtask.agent, True, text)
    except LLMError as err:
        return AgentResult(subtask.id, subtask.agent, False, str(err))
