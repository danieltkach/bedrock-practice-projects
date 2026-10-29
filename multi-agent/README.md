# Multi-Agent System (Amazon Bedrock)

A command-line assistant that answers complex questions by **orchestrating several specialized models**.
A cheap *coordinator* reads the question and writes a plan; specialist agents (analysis, writing, code)
work on the parts, in parallel when they can; a strong *aggregator* merges the results into one coherent
answer. Every run prints a cost and latency report.

## Features

| Feature | How |
|---|---|
| Coordinator agent | Claude Haiku 4.5 returns a JSON plan: subtasks, agent per subtask, dependencies |
| Specialized agents | `analysis` (Sonnet 4.5), `writing` (Haiku 4.5), `code` (Sonnet 4.5), each with its own system prompt |
| Intelligent routing | Three routes derived from the plan: **direct** (no specialist), **single** (one agent, no merge), **multi** (several agents + merge) |
| Dependencies and parallelism | Independent subtasks run at the same time (thread pool); a dependent subtask waits and receives the earlier outputs |
| Result aggregation | Claude Sonnet 4.5 merges the outputs, removes repetition, reports any missing part |
| Cost and latency optimization | Cheap model for cheap work; aggregator skipped when there is one agent; per-call tokens, seconds, estimated dollars; comparison with "everything on the strong model" |
| Reliability | Retry with exponential backoff and jitter; plan validation with safe fallback; one failed agent does not kill the answer |
| Observability | Routing decisions and every call are logged to `multi-agent.log` (`--verbose` shows them live) |
| Offline mode | `--mock` fake client and 19 unit tests, no AWS needed |

## Architecture

```
                         user question
                               |
                               v
                  +---------------------------+
                  | COORDINATOR (Haiku 4.5)   |  returns JSON plan, validated by Python
                  +---------------------------+  (unknown agent / cycle / too many -> safe fallback)
                    |            |            |
          no subtasks      1 subtask     2+ subtasks
             "direct"       "single"        "multi"
                |              |              |
        coordinator's     one agent      wave 1 (parallel)   wave 2 (needs wave 1)
        short answer        answer       +-----------+       +-----------+
                |              |         | analysis  |------>| writing   |
                |              |         | (Sonnet)  |       | (Haiku)   |
                |              |         +-----------+       +-----------+
                |              |         | code      |             |
                |              |         | (Sonnet)  |-------------+
                |              |         +-----------+             |
                |              |                 \                 /
                |              |                  v               v
                |              |            +----------------------+
                |              |            | AGGREGATOR (Sonnet)  |  one coherent answer
                |              |            +----------------------+
                v              v                        |
              +------------------------------------------------+
              | answer + routing line + cost/latency report    |
              +------------------------------------------------+
```

Every box that calls a model goes through `llm.py` (Converse API, retry, token/latency tracking).

## Quick start

Prerequisites: Python 3.10+, AWS CLI v2.32+, an AWS account with Bedrock access in `us-east-1` (Claude
models need the one-time Anthropic use-case form, shown in the Bedrock console the first time you open a
Claude model in the playground).

```bash
./run.sh setup                           # virtualenv + dependencies
./run.sh test                            # 19 offline tests, no AWS needed
./run.sh demo                            # whole pipeline with a fake client, no AWS needed

aws login --remote --region us-east-1    # or any other way to get AWS credentials
./run.sh test-live                       # real end-to-end check (a few cents)
./run.sh chat                            # interactive assistant
./run.sh ask "Compare SQS and Kinesis, then write an email to my team"
```

Direct use (after `./run.sh setup`): `.venv/bin/python main.py "question"`, `--mock`, `--verbose`,
`--no-report`, `--examples`.

### Example session (real run, abbreviated)

```
$ ./run.sh ask "Analyze the pros and cons of moving a monolith to microservices, then write a one-paragraph executive summary for non-technical managers."

[routing] multi: s1=analysis, s2=writing(after s1)
[why] This requires analysis of trade-offs first, then writing a summary based on that analysis.
...final merged answer...
role        model           in tok out tok    sec tries     est $
coordinator claude-haiku-      348     250   3.11     1   0.00160
analysis    claude-sonnet      183     804  20.49     1   0.01261
writing     claude-haiku-      981     273   4.47     1   0.00235
aggregator  claude-sonnet     1340    1083   8.90     1   0.02026
Total estimated cost: $0.03682   (same tokens all on the strong model: $0.04471, saving 18%)
```

`./run.sh chat` also has `/examples` (five ready-made queries covering all routes) and `/quit`.

## Project layout

```
main.py            CLI: single-shot and chat, routing line, report
config.py          every knob: models per role, token budgets, limits, retry, price table
llm.py             the only place that calls Bedrock: Converse, retry/backoff, text-only parsing
coordinator.py     planner prompt, plan parsing and validation, fallback
agents.py          system prompts of the specialists, aggregator and direct answerer; run_agent
orchestrator.py    routes, dependency waves, thread pool, aggregation, answer_query
metrics.py         per-call records, estimated cost, report
mock_client.py     fake bedrock-runtime client (offline demo and tests)
live_test.py       real end-to-end check (./run.sh test-live)
tests/test_all.py  19 offline tests
run.sh             single entry point
STUDY_GUIDE.md     concepts, walkthrough, traced example, rehearsal Q&A
```

## Configuration

Everything is in `config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `ROLE_MODEL` | coordinator/direct/writing: Haiku 4.5; analysis/code/aggregator: Sonnet 4.5 | model tier per role (the cost/quality dial) |
| `ROLE_MAX_TOKENS` | 500 to 3500 | output budget per role; a truncated answer logs a warning |
| `MAX_SUBTASKS` | 5 | plans with more are rejected (cost guard) |
| `MAX_PARALLEL` | 4 | threads for independent subtasks |
| `MAX_ATTEMPTS`, `BACKOFF_*` | 5, 1 s base, 20 s cap | retry policy for throttling and transient errors |
| `PRICES_PER_MILLION` | Sonnet 3/15, Haiku 1/5, Nova Lite 0.06/0.24 USD | **estimates typed by hand: verify at aws.amazon.com/bedrock/pricing** |
| `BASELINE_MODEL` | Sonnet 4.5 | the "everything on one strong model" comparison |

## Design decisions

- **Plan is data, validated by code.** The model understands the question; Python enforces rules (known
  agents, unique ids, no cycles, at most 5 subtasks). Bad plan -> fallback to one analysis agent. The route
  is *derived* from the number of subtasks, so the model cannot contradict itself.
- **Three routes.** Not every question deserves a committee. Direct and single routes skip the aggregator,
  the most expensive call.
- **Waves for parallelism.** Subtasks with satisfied dependencies run together; the wall-clock time of a wave
  is its slowest member. Threads fit because the work is waiting on the network.
- **Specialists never raise.** A failure becomes a result flagged `UNAVAILABLE`; the aggregator tells the
  user which part is missing, so partial answers survive throttling or an access error.
- **We own retries.** boto3's own retries are set to 1 attempt so backoff is visible in the log and report.
- **Text-only parsing.** Only content blocks with `text` are joined (some models return reasoning blocks).
- **Claude 4.5 family only.** Claude 5.x models think first and can burn a small token budget with no text.
- **Public/no customer data.** Example queries are generic; no secrets in code; credentials come from `aws login`.

## Known limitations and suggested improvements

| Limitation seen in testing | Improvement |
|---|---|
| Savings vs all-Sonnet were modest (11-18% in live runs) because Sonnet still produces most tokens (analysis, code, merge) | Use Haiku for more roles and measure quality; send the aggregator only summaries; try Nova Lite for the planner |
| The aggregator re-reads and re-writes everything: it is the slowest/most expensive step (about 9-19 s) | Stream output; skip the merge when outputs are already complementary; have specialists write shorter outputs |
| Whole-pipeline latency is 15-45 s (sequential waves + long outputs) | Streaming (`converse_stream`), lower `maxTokens`, prompt caching of the system prompts |
| Planner quality varies with the model; the planning prompt is the weak link | Few-shot examples in the prompt; a regression set of questions with expected routes |
| Prices are a hand-typed table | Read prices from the AWS Price List API |
| No quality evaluation of answers | LLM-as-judge or a rubric test set comparing multi-agent vs single-model answers |
| Agents cannot use tools (search, run code) | Converse `toolConfig` tool use; a reviewer/critic agent that checks code |
| In-memory chat history only | Persist sessions; summarize old turns |
| Local process only | Step Functions or AgentCore for deployment; Bedrock Guardrails for input/output safety |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Your session has expired` / no credentials | `aws login --remote --region us-east-1` (sessions last up to 12 h) |
| `Using the login credential provider requires ... botocore[crt]` | already in `requirements.txt`; re-run `./run.sh setup` |
| `Model use case details have not been submitted` | open a Claude model once in the Bedrock playground and complete the form; wait a few minutes |
| `NoRegionError` | `export AWS_DEFAULT_REGION=us-east-1` (run.sh sets it) |
| `ThrottlingException` | retried automatically (up to 5 times); if it persists lower `MAX_PARALLEL` or request a quota increase |
| `no text returned (stopReason=max_tokens)` | a model spent its whole budget (typical for Claude 5.x); use a 4.5 model or raise `ROLE_MAX_TOKENS` |
| Plan shows `[fallback plan]` | the coordinator returned invalid JSON/plan; see `multi-agent.log` for the reason |
