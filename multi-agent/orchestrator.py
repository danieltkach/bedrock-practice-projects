"""Runs a plan: direct / single / multi routes, parallel waves, then aggregation."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import config
from agents import AGGREGATOR_PROMPT, DIRECT_PROMPT, run_agent
from coordinator import make_plan
from llm import LLMError

log = logging.getLogger("multi-agent")


@dataclass
class QueryResult:
    answer: str
    plan: object
    results: dict            # subtask id -> AgentResult
    wall_seconds: float


def describe_plan(plan):
    """One readable line, e.g. 'multi: s1=analysis, s2=code, s3=writing(after s1,s2)'."""
    if not plan.subtasks:
        return "direct: answered by the coordinator tier, no specialists"
    parts = []
    for s in plan.subtasks:
        after = f"(after {','.join(s.depends_on)})" if s.depends_on else ""
        parts.append(f"{s.id}={s.agent}{after}")
    note = "  [fallback plan]" if plan.fallback else ""
    return f"{plan.route}: " + ", ".join(parts) + note


def split_into_waves(subtasks):
    """Group subtasks into waves. Everything inside a wave is independent -> runs in parallel.
    A subtask enters a wave only after all of its dependencies finished in earlier waves."""
    done, remaining, waves = set(), list(subtasks), []
    while remaining:
        wave = [s for s in remaining if all(d in done for d in s.depends_on)]
        if not wave:   # cannot happen after validation; guard against an endless loop anyway
            raise ValueError("unresolvable dependencies")
        waves.append(wave)
        done.update(s.id for s in wave)
        remaining = [s for s in remaining if s.id not in done]
    return waves


def execute_plan(llm, plan, query):
    results = {}
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL) as pool:
        for wave in split_into_waves(plan.subtasks):
            futures = [
                pool.submit(run_agent, llm, s, query, [results[d] for d in s.depends_on])
                for s in wave
            ]
            for future in futures:
                res = future.result()
                results[res.subtask_id] = res
    return results


def aggregate(llm, query, plan, results):
    """Merge several agent outputs into one answer with the strong model."""
    sections = []
    for s in plan.subtasks:
        res = results[s.id]
        body = res.text if res.ok else f"UNAVAILABLE ({res.text})"
        sections.append(f"### Result of the {s.agent} agent for: {s.task}\n{body}")
    prompt = f"Original user question:\n{query}\n\n" + "\n\n".join(sections)
    try:
        return llm.call("aggregator", AGGREGATOR_PROMPT, prompt)
    except LLMError as err:
        # Last resort: still give the user the raw material instead of nothing.
        log.error("aggregator failed: %s", err)
        good = [results[s.id].text for s in plan.subtasks if results[s.id].ok]
        return "(Could not merge the results; showing them separately.)\n\n" + "\n\n".join(good)


def answer_query(llm, query, history_text=""):
    """The whole pipeline for one question."""
    start = time.perf_counter()
    plan = make_plan(llm, query, history_text)
    results = {}

    if plan.route == "direct":
        if plan.direct_answer:
            answer = plan.direct_answer                       # the coordinator already answered
        else:
            try:
                answer = llm.call("direct", DIRECT_PROMPT, query)
            except LLMError as err:
                answer = f"Sorry, I could not answer: {err}"
    else:
        results = execute_plan(llm, plan, query)
        if plan.route == "single":
            only = results[plan.subtasks[0].id]
            # One agent = nothing to merge, so skip the (expensive) aggregator call.
            answer = only.text if only.ok else f"Sorry, the {only.agent} agent failed: {only.text}"
        elif not any(r.ok for r in results.values()):
            answer = "Sorry, every specialist agent failed. Check credentials/throttling and retry."
        else:
            answer = aggregate(llm, query, plan, results)

    return QueryResult(answer, plan, results, time.perf_counter() - start)
