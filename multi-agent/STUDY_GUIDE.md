# Study Guide: how the Multi-Agent System works

Goal: after reading this you can explain every part, why it was built that way, what went wrong, and what
you would improve. Read in order.

---

## 1. The problem and the idea (60-second version)

**Problem.** One model call is not always the right tool. A question like *"compare SQS and Kinesis,
write a Python consumer for the better one, and draft an email to my team"* mixes reasoning, code and
writing. Sending it all to the biggest model works but is slow and costly; sending it all to the smallest
model risks poor reasoning and code.

**Idea: orchestration.** Split the work like a small team:

1. A **coordinator** reads the question and makes a plan (who does what, in which order).
2. **Specialists** do their parts, at the same time when they don't depend on each other.
3. An **aggregator** merges the pieces into one answer.
4. Cheap models do the easy parts, strong models the hard parts; we **measure** cost and time.

Analogy: a newsroom. The editor (coordinator) assigns stories, reporters (analysis, code) work in parallel,
a writer polishes after the reporters finish, and the editor-in-chief (aggregator) assembles the issue.

---

## 2. Glossary

| Term | Plain meaning | In this project |
|---|---|---|
| **Agent** | A model call wrapped with a role: system prompt + model choice | `analysis`, `writing`, `code` in `agents.py` |
| **Coordinator / orchestrator** | The part that decides who does what and runs it | `coordinator.py` (decides), `orchestrator.py` (runs) |
| **Routing** | Choosing the path a question takes | direct, single, multi |
| **Subtask** | One unit of work assigned to one agent | `Subtask(id, agent, task, depends_on)` |
| **Dependency** | "B needs A's output first" | `depends_on: ["s1"]` |
| **Wave** | A group of subtasks that can run together | `split_into_waves` |
| **Aggregation** | Merging several outputs into one answer | `aggregate()` with Sonnet |
| **System prompt** | Standing instructions that define a model's role | `AGENT_PROMPTS` |
| **Converse API** | Bedrock's uniform chat interface across models | `client.converse(...)` in `llm.py` |
| **Token** | Piece of a word; models bill and limit in tokens | `usage.inputTokens/outputTokens` |
| **maxTokens** | Output limit for one call | `ROLE_MAX_TOKENS` |
| **Throttling** | AWS saying "too many requests, slow down" | `ThrottlingException` |
| **Exponential backoff** | Wait 1 s, 2 s, 4 s... between retries | `LLM._backoff` |
| **Jitter** | Small random extra wait so parallel callers don't retry at the same instant | `random.uniform` |
| **Thread pool** | Several tasks running at once; good when waiting on the network | `ThreadPoolExecutor` |
| **Latency** | How long something takes | seconds per call and per query |
| **Fallback** | A safe plan B when something is invalid | one analysis agent answers everything |
| **Mock / fake client** | A stand-in object that behaves like the real one, for tests | `mock_client.py` |

---

## 3. The flow, step by step

`answer_query` in `orchestrator.py`:

1. **Plan** (`make_plan`): Haiku gets the question (plus up to 3 recent Q/A in chat mode) and must reply
   with JSON only. `parse_plan` validates: known agents, non-empty tasks, unique ids, dependencies exist,
   no cycles, at most 5 subtasks. Anything wrong -> fallback plan (one analysis agent).
2. **Route** (derived from the number of subtasks):
   - **direct**: no subtasks. The coordinator's own short answer is returned (1 model call in total).
   - **single**: one subtask. That agent's text is returned as is. No aggregator: nothing to merge, and the
     merge is the priciest call.
   - **multi**: two or more subtasks.
3. **Execute** (`execute_plan`): subtasks are grouped into waves. In each wave all subtasks start at once in a
   thread pool. A later subtask receives the earlier outputs in its prompt (`build_agent_prompt`).
4. **Aggregate** (`aggregate`): Sonnet gets the question plus every result and writes one answer. Failed parts
   arrive marked `UNAVAILABLE`.
5. **Report**: `Tracker.report` prints tokens, seconds, attempts and estimated dollars per call, the total, the
   "all on Sonnet" comparison, and real time vs sum of call times (that gap is the parallel saving).

---

## 4. Code walkthrough

- **`config.py`**: constants only. `ROLE_MODEL` is the cost/quality dial; `PRICES_PER_MILLION` is a table of
  *estimates*.
- **`llm.py`**: `LLM.call` builds the Converse request (`modelId`, `system`, `messages`, `inferenceConfig`),
  loops up to 5 attempts, sleeps with backoff only for retryable error codes, records a `CallRecord` and
  returns text. `extract_text` joins only blocks that contain `"text"`.
- **`coordinator.py`**: `planner_system_prompt` lists the agents and the JSON format. `_extract_json` copes
  with ```json fences. `parse_plan` validates. `_has_cycle` is a depth-first search. `Plan.route` is a property
  computed from the subtasks.
- **`agents.py`**: prompts for each role, `build_agent_prompt`, and `run_agent`, which catches `LLMError` and
  returns `AgentResult(ok=False)` instead of raising.
- **`orchestrator.py`**: `split_into_waves`, `execute_plan`, `aggregate`, `answer_query`.
- **`metrics.py`**: `Tracker` (with a lock, because threads add records at once), `cost_usd`, `report`.
- **`mock_client.py`**: fake `converse` that detects the role from the system prompt; keyword-based planner.
- **`main.py`**: argparse, chat loop, logging setup, real client creation with `retries max_attempts=1`.
- **`tests/test_all.py`**: 19 tests using a scripted fake client (queued responses/errors) and the mock.

**Python notes.** `@dataclass` makes a simple record class. `ThreadPoolExecutor.submit(f, args)` starts `f`
in a thread and returns a future; `.result()` waits for it. `with lock:` makes only one thread touch shared data
at a time. `{**a, **b}` merges dictionaries.

---

## 5. A real example, traced

Question: *"Analyze the pros and cons of moving a monolith to microservices, then write a one-paragraph
executive summary for non-technical managers."*

| Step | What happened | Time | Est. cost |
|---|---|---|---|
| Coordinator (Haiku) | Plan: `s1=analysis`, `s2=writing` depends on `s1`. Reason: "writing depends on the analysis output" | 3.1 s | $0.0016 |
| Wave 1: analysis (Sonnet) | 804 output tokens of pros/cons | 20.5 s | $0.0126 |
| Wave 2: writing (Haiku) | Got the analysis in its prompt (981 input tokens), wrote the summary | 4.5 s | $0.0024 |
| Aggregator (Sonnet) | Merged both into one document | 8.9 s | $0.0203 |
| **Total** | | **37 s** | **$0.037** (all-Sonnet: $0.045, saving 18%) |

The dependency made this run sequential, so parallelism saved nothing here. A second example (compare SQS
and Kinesis + code + email) planned `s1=analysis`, `s2=code`, `s3=writing(after s1)`; analysis and code ran
together: 45 s real vs 64 s if sequential (18.6 s saved).

---

## 6. Design decisions and trade-offs (expect "why did you...?")

| Decision | Why | Trade-off |
|---|---|---|
| Haiku as coordinator | Planning is short structured output; cheap and fast | A weaker planner may route badly (mitigated by validation + fallback) |
| Sonnet for analysis, code, aggregator | Quality matters most there | Most of the cost and time is spent on Sonnet |
| Haiku for writing | It reformats already-reasoned material | May be less polished; the aggregator smooths it |
| Plan as JSON validated in Python | Models can't be trusted blindly; rules are cheap to enforce | Extra code |
| Route derived from subtasks | Impossible to have route="single" with 3 subtasks | None worth mentioning |
| Skip aggregator for one agent | Saves the most expensive call | None: nothing to merge |
| Threads, not async | Simple for beginners; calls wait on the network | Async scales further |
| We own retries (boto3 retries = 1) | Backoff is visible in logs/report | We maintain the code |
| Estimated price table in config | Simple, editable | Can go stale: say "estimate" and verify |
| Fake client + tests | Works without credentials; fast; safe | Mock plans use keywords, not real model behavior (so `test-live` exists) |

---

## 7. How it was built with AI

1. **Read the requirement** (coordinator, 3 specialists, routing, aggregation; optimize cost and latency) and
   decomposed it: LLM wrapper -> metrics -> agents -> coordinator -> orchestrator -> CLI.
2. **Reused lessons from the previous project** (Project 1: `botocore[crt]` for `aws login`, text-only parsing,
   avoid Claude 5.x for small budgets, public data only) so known problems were designed out.
3. **Built a fake client first** so every piece could be tested with no AWS access, then added the real one.
4. **Wrote tests for the risky logic**: malformed plans, cycles, throttling, failing agents, parallelism.
5. **Ran live** and read the real report; that revealed truncated outputs (see section 8).
Key message: *the AI writes and explains code; a human sets requirements, reads the evidence (the report and
logs) and decides trade-offs.*

---

## 8. Things that went wrong (and what each taught)

1. **Truncated answers.** First live run: the code agent hit exactly its 1,400-token budget and the aggregator
   its 1,800. Fix: bigger budgets (2,500 / 3,500) and a log warning on `stopReason == max_tokens`. Lesson:
   budgets are a cost guard *and* a quality risk.
2. **Savings smaller than expected.** Only 11-18% cheaper than all-Sonnet, because Sonnet still writes most of the
   tokens. Lesson: measure before claiming; real savings need more roles on cheap models or shorter strong-model output.
3. **Latency dominated by long outputs and by dependencies.** Lesson: parallelism only helps independent work.
4. **Known pitfalls carried over:** Claude 5.x models can return no text with small budgets; credentials from
   `aws login` need `botocore[crt]` and expire.

---

## 9. Likely questions, with short answers

1. **What is a multi-agent system?** Several model calls with different roles and prompts, coordinated to answer one request.
2. **What does the coordinator do?** Classifies the question and outputs a JSON plan; Python validates it.
3. **What are the routes?** Direct (no specialist), single (one agent, no merge), multi (several agents + aggregator).
4. **How do you run things in parallel?** Group subtasks into dependency waves; run each wave in a thread pool.
5. **What if the model returns a bad plan?** Validation catches unknown agents, cycles, duplicates, too many; fallback to one analysis agent.
6. **What if one agent fails?** It returns an `UNAVAILABLE` result; the others finish; the aggregator tells the user what is missing.
7. **How do you handle throttling?** Retry up to 5 times with exponential backoff and jitter, only for retryable error codes.
8. **How do you optimize cost?** Cheap model for planning/writing, strong for reasoning/code/merge, skip the aggregator when not needed, cap subtasks and tokens, measure with the report.
9. **How do you optimize latency?** Parallel waves, token limits, fewer calls; improvements: streaming, caching.
10. **How accurate is the cost report?** Token counts are exact from the API; dollars use a hand-typed price table, so they're estimates to verify.
11. **Why a stronger model for aggregation?** It decides coherence of the final answer; errors there are visible to the user.
12. **Why not use Bedrock Agents / a framework?** For learning: seeing the loop explicitly makes the logic explainable; frameworks (Bedrock Agents, AgentCore, LangGraph, Strands) are the next step for production.
13. **How did you test without AWS?** A fake client implementing `converse` (scripted responses and errors) plus a mock mode; and `test-live` for the real thing.
14. **Security?** No secrets in code (short-lived `aws login` credentials), no customer data, least-privilege IAM (`bedrock:InvokeModel` on the used models) in production, Guardrails as an improvement.
15. **When is multi-agent NOT worth it?** Simple questions: the extra calls cost more time and money than one good call. That's why direct/single routes exist.
16. **How would you evaluate answer quality?** A question set with a rubric; LLM-as-judge comparing multi-agent to a single strong-model answer.

---

## 10. Hands-on exercises

1. `./run.sh demo`, then read `multi-agent.log` to see the routing decision.
2. In `config.py` set `ROLE_MODEL["writing"]` to Sonnet and run the same query: compare cost and style.
3. Set `MAX_PARALLEL = 1` and see the latency line change.
4. Ask a trivial question ("hello") and a code-only question: confirm direct and single routes skip the aggregator.
5. Add a fourth agent, e.g. `research` or `review`: edit `AGENT_PROMPTS`, `AGENT_DESCRIPTIONS`, `ROLE_MODEL`, `ROLE_MAX_TOKENS`; add a test.
6. Change a price in `PRICES_PER_MILLION` and watch the report.
7. Break the planner on purpose (change "ONLY a JSON object") and see the `[fallback plan]` path.
