# Evaluation Platform (Amazon Bedrock)

A command-line benchmarking system that **compares several Bedrock models on the same test suite** and
produces an overall ranking, per-category winners, automatic recommendations, a Markdown report and an
interactive HTML dashboard. It measures three things for every model: **quality** (automatic checks plus
an LLM judge), **cost** (real token counts times an editable price table) and **latency**.

## Features

| Feature | How |
|---|---|
| 6 models side by side (7 with opt-in Opus) | Claude Sonnet 4.5, Claude Haiku 4.5, Nova Pro, Nova 2 Lite, Nova Lite, Nova Micro; Opus 4.6 is opt-in because it is expensive |
| 25 test categories, 59 tests | Plain JSON files in `testdata/` (one file per category). Invented or public content only |
| Automatic checks | 22 check types: exact, contains, regex, numeric answer, JSON fields, order constraints, refusal, **run generated Python against hidden asserts, run generated SQL on an in-memory SQLite database, test generated regexes against match/no-match strings** |
| LLM-as-judge | Rubric, score 1-5 plus a one-sentence reason, JSON output, temperature 0, blind to the author. Primary judge Sonnet 4.5; optional second judge; a model is never judged by itself when a second judge exists |
| Aggregate score | Weighted quality + cost + latency (defaults 0.6 / 0.2 / 0.2; CLI flags `--w-quality --w-cost --w-latency`) |
| Automatic recommendations | Best overall, highest quality, best value, fastest acceptable, best per category group. Every line shows its rule and evidence; no LLM involved (an optional LLM narrative is clearly marked) |
| Cost control | Hard budget cap checked **before** every call, `--limit`, `--models`, `--categories`, `--dry-run`, on-disk response cache (re-runs are free) |
| Concurrency and reliability | Thread pool, per-model concurrency limit, retry with exponential backoff and jitter, structured logging, resumable JSONL results, failures recorded instead of crashing the run |
| Interactive dashboard | ONE self-contained HTML file (inline JS/CSS/SVG, no CDN, no fetch): model and category/group filters, live weight sliders, sortable tables, scatter, latency bars, stacked score bars, category heatmap, drill-down to each prompt/answer/check/judge reason |
| Offline mode | `--mock` fake client and 62 unit tests, no AWS needed |
| Zero extra dependencies | Only `boto3` and `botocore[crt]`, same pins as the other projects |

## Architecture

```
 testdata/*.json  (25 categories, 59 tests: prompt, checks, rubric, reference)
        |
        v
 main.py run  --models --limit --budget --dry-run
        |
        |  runner.py: tasks = (model x test), interleaved; thread pool (8) + max 2 calls per model
        v
 +------------------------------ one (model, test) pair ------------------------------+
 |  llm.py  -> cache hit? -> budget check -> Converse API (retry/backoff) -> text only  |
 |      |                                                                               |
 |      +--> checks.py  automatic score 0..1 (regex, numeric, python/SQL/regex execution)|
 |      +--> judge.py   judge model(s) score 1..5 against the rubric (never self-judge)  |
 +--------------------------------------------------------------------------------------+
        |   one JSON line appended as soon as the pair finishes  (resumable)
        v
 results/results.jsonl + meta.json
        |
        v
 scoring.py   quality per answer -> category means -> model means; normalize cost and latency;
        |     weighted aggregate;  recommend.py: transparent rules -> recommendations + warnings
        +--> report.py     -> report.md          (tables, winners, heatmap, judge agreement, caveats)
        +--> dashboard.py  -> dashboard.html     (data embedded as JSON; JS recomputes live)
```

Every model call (test answers, judge, optional narrative) goes through `llm.py`.

## Quick start

Prerequisites: Python 3.10+ (developed on 3.14), AWS CLI v2.32+, an AWS account with Bedrock access in
`us-east-1` for the models in `config.py` (Claude models need the one-time Anthropic use-case form).
The code-execution checks use the `resource` module, so run on Linux/macOS/WSL.

```bash
./run.sh setup                  # virtualenv + dependencies
./run.sh test                   # 62 offline tests, no AWS needed
./run.sh demo                   # whole pipeline with a fake client (free) -> demo_output/

# Look at the real results without spending anything:
#   sample_output/report.md  and  sample_output/dashboard.html (open it in a browser)

aws login --remote --region us-east-1      # or any other way to get AWS credentials
./run.sh test-live              # tiny real end-to-end check (a few cents)
./run.sh plan                   # dry run: planned calls and estimated cost, nothing is called
./run.sh run --limit 1 --budget 1          # cheap live run: 1 test per category
./run.sh run --judge2 opus-4.6             # full live run with a second judge (what sample_output used)
./run.sh report --w-quality 0.8 --w-cost 0.1 --w-latency 0.1     # re-weight, free
./run.sh dashboard              # rebuild report.md + dashboard.html from saved results, free
```

Direct use after setup: `.venv/bin/python main.py <command>`. Commands: `list`, `validate`, `run`,
`rescore`, `report`, `dashboard`. Useful `run` flags:

| Flag | Meaning |
|---|---|
| `--models a,b` / `--models all` | short names from `python main.py list` (default: the 6 default models; `opus-4.6` is opt-in) |
| `--categories x,y` | only these categories |
| `--limit N` | at most N tests per category |
| `--budget USD` | hard cap on estimated spend (default 3.00); the run stops before exceeding it and keeps partial results |
| `--dry-run` | print planned calls and estimated cost, call nothing |
| `--judge M`, `--judge2 M\|none`, `--no-judge` | choose judge models, or use automatic checks only |
| `--no-exec` | do not execute generated code (Python and regex checks are skipped) |
| `--no-cache` | ignore the response cache (costs money again) |
| `--out DIR` | results folder (default `results/`; `demo_output/` with `--mock`) |
| `--mock` | fake client: no AWS, no cost |

An aborted or crashed run is **resumed** by running the same command again: finished pairs are skipped.

## What the live run found (single small sample, be careful)

One run on 2026-10-27, `us-east-1`, 6 models x 59 tests = 354 answers, judges Sonnet 4.5 + Opus 4.6
(files in `sample_output/`). Cost, latency and judge costs use the estimated prices in `config.py`.

| # | Model | Aggregate | Quality (95% interval) | Cost / call | Median latency | p95 |
|---|---|---|---|---|---|---|
| 1 | Nova Lite | 0.921 | 0.910 (+/-0.081) | $0.00004 | 0.82 s | 2.9 s |
| 2 | Nova Micro | 0.909 | 0.848 (+/-0.087) | $0.00003 | 0.78 s | 2.3 s |
| 3 | Claude Haiku 4.5 | 0.786 | 0.983 (+/-0.015) | $0.00087 | 1.31 s | 4.1 s |
| 4 | Nova 2 Lite | 0.780 | 0.908 (+/-0.067) | $0.00052 | 1.14 s | 4.0 s |
| 5 | Nova Pro | 0.776 | 0.884 (+/-0.078) | $0.00055 | 1.01 s | 3.4 s |
| 6 | Claude Sonnet 4.5 | 0.594 | 0.990 (+/-0.009) | $0.00255 | 2.86 s | 7.8 s |

Automatic recommendations from that run: **best overall** Nova Lite (it wins because cost and latency
get 40% of the weight and it is cheap and fast with decent quality); **highest quality** Sonnet 4.5;
**best value** Nova Lite (cheapest model with at least 90% of the best quality); **fastest acceptable**
Nova Micro; best for code Nova Lite; for knowledge and language Sonnet 4.5; for reasoning and for safety
Haiku 4.5 (tied with Sonnet).

What to take from it:

- **The weights decide the winner.** With quality only, the order is Sonnet 0.990, Haiku 0.983, Nova Lite
  0.910. With weights 0.8/0.1/0.1 the order is Nova Lite 0.916, Haiku 0.884, Nova Micro 0.879. Use the
  dashboard sliders.
- **Where the models really differ:** *hallucination resistance* (asked to summarize an invented paper, both
  Claude models said they could not verify it while all four Nova models confidently invented findings;
  the invented-town question was handled by every model), *ambiguity handling* (Claude asked a clarifying question; Nova models mostly wrote a
  generic template), *summarization under length limits*, and hard multi-step arithmetic/logic (Nova
  Micro and Nova Pro got a 5-digit x 4-digit multiplication wrong).
- **Where they do not differ:** 10 of 25 categories were solved perfectly by all 6 models and 12 by all
  models at 0.9 or more. The suite is partly too easy to separate the top models (a ceiling effect).
- **Quality gaps among the top models are within noise** (intervals overlap); the large effects are
  cost (Sonnet is about 60x Nova Lite per call) and latency (about 3.5x).
- **The judge matters.** In an earlier run with Haiku 4.5 as second judge, Haiku (0.990) ranked above
  Sonnet (0.980), because the Haiku judge was stricter (mean 3.79 vs 4.04 on 76 doubly judged answers)
  and was the *only* judge of Sonnet's answers. With Opus 4.6 as second judge Sonnet is on top. The
  two judges in the final run agreed exactly 66% of the time (mean difference 0.35 points).

**Honest limits:** 59 tests, one run, one region, latency measured once under 2-way per-model
concurrency, prices are estimates, a judge is itself a model. Treat the numbers as a demonstration of the
method and a first filter, not a verdict. Cost to reproduce the whole sample from scratch: about $0.96
at the estimated prices; cash actually spent producing it, including a pilot and reruns after fixing the
test data: about $1.22 (an estimate, not a bill). Note: in the sample every record is flagged "served from cache"
because the final pass reused the earlier paid calls; latencies are those of the first, real call.

## Project layout

```
main.py                CLI: list, validate, run, rescore, report, dashboard
config.py              every knob: models and prices, judge, weights, budget, retry, rules
llm.py                 the only place that calls Bedrock: Converse, retry/backoff, cache, budget check
cache.py               on-disk response cache keyed by (model, system, prompt, params)
budget.py              hard spending cap, thread-safe
runner.py              concurrent run, JSONL results, resume, dry-run planning, rescore
checks.py              22 automatic check types
sandbox.py             restricted subprocess for generated Python (best effort, see Limitations)
judge.py               judge prompt, JSON parsing with fallback, "never judge yourself" logic
scoring.py             quality per answer, normalization, aggregate score, category winners, judge agreement
recommend.py           transparent recommendation rules and warnings
report.py              Markdown report (+ optional LLM narrative, marked as generated)
dashboard.py           injects data into dashboard_template.html
dashboard_template.html  all dashboard JS/CSS/SVG (one file, no libraries)
mock_client.py         fake bedrock-runtime client (offline demo and tests)
testdata.py            loads and validates the JSON test files
testdata/*.json        25 category files: the test suite
live_test.py           tiny real end-to-end check (./run.sh test-live)
tests/test_all.py      62 offline tests
sample_output/         real results of the live run: report.md, dashboard.html, results.jsonl, meta.json
run.sh                 single entry point
STUDY_GUIDE.md         concepts, walkthrough, traced example, rehearsal Q&A
```

## The test suite

25 categories, 2 to 3 tests each (59 in total). Every test stores a `reference` answer, and
`python main.py validate` checks that **each reference passes its own automatic checks**, which catches
typos in the data.

| Group | Categories |
|---|---|
| reasoning | arithmetic, logic_puzzles, math_word_problems, multi_step_planning |
| code | code_generation, code_explanation, code_debugging, sql_generation, regex_writing |
| language | summarization, translation, tone_rewriting, legal_clause_explanation (invented clauses) |
| structured | instruction_following, json_extraction, classification, table_qa, data_formatting |
| knowledge | factual_qa, long_context_needle (about 1,500 to 2,500 word documents), common_sense, ambiguity_handling |
| safety | safety_refusal, over_refusal (harmless requests that sound scary), hallucination_resistance |

**Add a test:** open a file in `testdata/`, append an object to `tests` with `id`, `prompt`, `checks`
and/or `rubric` (+ `judge: true`), a `reference` answer, optionally `max_tokens`. Run
`python main.py validate`. **Add a category:** add a new JSON file with a `category`, a `group`
(see `GROUP_TITLES` in `config.py`) and its tests. **Add a check type:** write a function in `checks.py`
and register it in `CHECKS`.

## How the score is built

1. **Answer quality (0 to 1).** Automatic checks give 0..1 (average of the test's checks; code, SQL and regex
   checks give the fraction of cases passed). The judge gives 1-5, mapped to 0..1. A test with both uses 50/50
   (`AUTO_WEIGHT`). Empty answers count 0; API failures are excluded (not the model's fault) and counted in
   "Failed calls".
2. **Model quality** = mean over categories of the category mean (each category counts equally).
3. **Cost** = mean estimated USD per test call; **latency** = median seconds.
4. **Normalization** across the models being compared: quality as is (already 0..1); cost by min-max on a
   logarithmic scale (prices span 100x); latency by min-max. Best = 1, worst = 0.
5. **Aggregate** = `wq*quality + wc*cost_score + wl*latency_score`, weights normalized to sum 1.
6. **Recommendations** (`recommend.py`): *best value* = cheapest model with quality at least 90% of the best;
   *fastest acceptable* = lowest median latency with quality at least 85% of the best; *best for group* =
   highest mean quality in the group, near-ties (0.02) go to the cheaper model. Warnings appear for
   near-ties, overlapping intervals, failures, truncated answers and self-judged answers.

## Configuration

Everything is in `config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `MODELS` | 6 default + Opus 4.6 opt-in | short name, inference profile id, label, **estimated** price per 1M tokens |
| `JUDGE_MODEL`, `SECOND_JUDGE_MODEL` | Sonnet 4.5, none | judges (CLI: `--judge`, `--judge2`) |
| `AUTO_WEIGHT` | 0.5 | automatic vs judge share when a test has both |
| `WEIGHTS` | quality 0.6, cost 0.2, latency 0.2 | aggregate score (CLI: `--w-quality` ...) |
| `VALUE_RATIO`, `FAST_RATIO`, `CATEGORY_TIE_MARGIN` | 0.90, 0.85, 0.02 | recommendation rules |
| `BUDGET_USD` | 3.00 | hard cap (CLI: `--budget`) |
| `MAX_WORKERS`, `PER_MODEL_CONCURRENCY` | 8, 2 | threads in total / simultaneous calls to one model |
| `DEFAULT_MAX_TOKENS`, `TEMPERATURE` | 700, 0.0 | per-test override with `max_tokens` in the JSON |
| `MAX_ATTEMPTS`, `BACKOFF_*` | 5, 1 s base, 20 s cap | retry policy for throttling and transient errors |
| `ENABLE_CODE_EXEC`, `CODE_TIMEOUT_SECONDS` | True, 5 | run generated code in the restricted subprocess |

**Prices are hand-typed estimates.** Verify them at https://aws.amazon.com/bedrock/pricing/ before trusting any
dollar figure.

## Design decisions

- **Results are data, scores are computed later.** `results.jsonl` stores the raw answer, tokens, latency and
  check/judge outcomes. Cost, weights and recommendations are recomputed at report time, so changing a price or a
  weight needs no new model calls.
- **Cache keyed by the full request, resume keyed by a test fingerprint.** Editing a prompt or token budget
  automatically invalidates the old result for that test; unchanged calls stay free.
- **Budget check before the call, with a worst case** (the model uses its whole output budget). The cap can be
  slightly conservative but is never exceeded by more than one in-flight call per thread. Cached answers cost nothing.
- **Tasks are interleaved** (test by test, all models) so that an aborted run still compares models fairly.
- **Judge safety:** blind, rubric, temperature 0, JSON; never judge a model with itself when another judge exists;
  report judge agreement; flag self-judged answers.
- **No BLEU/ROUGE/BERTScore.** They compare overlap with a reference text. Open-ended answers have many valid
  wordings, so a rubric judge plus executable checks is a better fit; adding them would need an extra
  dependency (BERTScore) for little value here.
- **Executable checks beat string matching** for code, SQL and regex: the generated artifact is actually run.
- **Text-only parsing** (only content blocks with `text`), and Claude 5.x is avoided because those models can
  spend a small token budget thinking and return no text.
- **Public/invented data only.** Contracts, papers and towns in the tests are invented. No secrets in code;
  credentials come from `aws login`.

## Known limitations and suggested improvements

| Limitation | Improvement |
|---|---|
| Small sample: 2-3 tests per category, one run, wide intervals | More tests per category (add to the JSON), repeat runs, bootstrap confidence intervals, report variance across runs |
| Ceiling effect: many categories are solved by every model | Add harder tests (the 9 hardest were added after the first run); use the "hardest tests" table to find where to dig |
| Judge bias and judge-set asymmetry (a model is judged by a different set of judges than the others) | Use 2-3 independent judges that are not candidates; calibrate against a human-labelled sample; pairwise comparisons instead of absolute scores |
| Latency measured once, under concurrency, mixed with network noise | Repeat each call, run sequentially for latency, record time-to-first-token with `converse_stream` |
| Fixed token budgets penalise verbose models (some Nova answers and one Sonnet answer were cut off) | Per-model budgets, or report "quality at equal output length" |
| Min-max normalization exaggerates small absolute differences (e.g. 0.8 s vs 1.3 s) | Ratio scores, or fixed reference values per use case ("under 2 s is a full score") |
| Prices are a hand-typed table | Read them from the AWS Price List API |
| Single-turn, text-only tests | Multi-turn chats, tool use, images, long documents, RAG scenarios |
| Code sandbox is best effort (allowed imports, empty environment, CPU/memory limits, timeout), not a security boundary | Run checks in a container or microVM; `--no-exec` disables them |
| Dashboard verified with a headless JS engine and a stub DOM, not in a real browser | Open `sample_output/dashboard.html` and check layout on your browsers and phone |
| Local process only | Step Functions / AWS Batch for large suites; Bedrock model evaluation jobs for managed alternatives |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No valid AWS credentials` | `aws login --remote --region us-east-1` (sessions expire) |
| `Using the login credential provider requires ... botocore[crt]` | already in `requirements.txt`; re-run `./run.sh setup` |
| `AccessDeniedException` / "use case details" error | enable the model in the Bedrock console and complete the Anthropic form once |
| `ThrottlingException` | retried automatically; lower `PER_MODEL_CONCURRENCY` or `--workers` |
| `ABORTED: budget cap ...` | results so far are saved; raise `--budget` and run again to resume |
| A model shows many "Failed calls" | open `eval-platform.log`; model not enabled, or no text returned (`max_tokens` too small for a thinking model) |
| "cut off by the token limit" warning | raise `max_tokens` for that test in the JSON (this re-runs only that test) |
| Edited a check, scores look old | `python main.py rescore` re-runs the automatic checks on saved answers, free |
| Edited a rubric | judge scores are stored; delete the results folder (the cache keeps the answers, so only judge calls are repeated) |
| Dashboard empty | open the file from disk in a modern browser with JavaScript enabled; run `report` first |
