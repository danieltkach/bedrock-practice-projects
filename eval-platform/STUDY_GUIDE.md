# Study Guide: how the Evaluation Platform works

Goal: after reading this you can explain every part, why it was built that way, what went wrong, and what
you would improve. Read in order.

---

## 1. The problem and the idea (60-second version)

**Problem.** There are many models on Bedrock (Claude Sonnet, Haiku, Nova Pro, Lite, Micro...), with prices that
differ by 100x. "Which one should I use?" is not answered by marketing pages or by trying three prompts by hand.
The right answer depends on the task, on how much quality you need, on cost and on speed.

**Idea: a repeatable benchmark.**

1. A **test suite**: many small tasks in 25 categories, stored as data (JSON).
2. A **runner** that asks every model every test, in parallel, and saves each result.
3. **Scoring**: *quality* (automatic checks, plus an LLM judge for open-ended answers), *cost* (tokens x price)
   and *latency* (seconds).
4. An **aggregate score** with weights you choose, per-category winners, and **automatic recommendations**.
5. A **report** (Markdown) and a **dashboard** (one HTML file) to explore the results.
6. **Cost control**, because the benchmark itself costs money: budget cap, cache, dry run.

Analogy: a driving test for models. Same route (tests), same examiner rules (checks and rubric), a stopwatch
(latency), a fuel gauge (cost), and in the end a ranking that depends on whether you value speed, economy or safety.

---

## 2. Glossary

| Term | Plain meaning | In this project |
|---|---|---|
| **Benchmark** | Same tests run on several systems so results can be compared | `main.py run` |
| **Test case** | One prompt plus how to judge the answer | an entry in `testdata/*.json` |
| **Category** | A group of tests of one skill (translation, SQL...) | one JSON file |
| **Automatic check** | Code that decides if an answer is right | `checks.py` |
| **Reference answer** | An example of a good answer | `reference` field |
| **Rubric** | What a good answer must do, written for the judge | `rubric` field |
| **LLM-as-judge** | A strong model scores another model's answer | `judge.py` |
| **Self-preference bias** | A judge tends to like answers that look like its own | never self-judge when another judge exists |
| **Blind judging** | The judge does not know which model wrote the answer | `build_judge_prompt` has no model name |
| **Quality** | Score 0..1 for an answer | `scoring.quality_of` |
| **Macro average** | Average of category averages, so every category weighs the same | `scoring.summarize` |
| **Normalization** | Rescaling a metric to 0..1 so metrics can be combined | `min_max_score` |
| **Min-max** | (value - worst) / (best - worst) | cost (log scale) and latency |
| **Aggregate score** | Weighted sum of quality, cost score, latency score | `WEIGHTS` |
| **Latency (median, p95)** | Typical time / time that 95% of calls beat | `latency_median`, `latency_p95` |
| **Token** | Piece of a word; models bill in tokens | `usage.inputTokens/outputTokens` |
| **Converse API** | Bedrock's uniform chat interface across models | `client.converse(...)` in `llm.py` |
| **Throttling / backoff / jitter** | "Too many requests"; wait 1, 2, 4 s...; random extra wait | `LLM._backoff` |
| **Thread pool** | Several tasks at once; good when waiting on the network | `ThreadPoolExecutor` in `runner.py` |
| **Cache** | Stored answers so a repeat costs nothing | `cache.py` |
| **Budget cap** | Hard limit on estimated spend | `budget.py` |
| **Dry run** | Plan and estimate without calling | `run --dry-run` |
| **JSONL** | One JSON object per line; easy to append and resume | `results.jsonl` |
| **Resume** | Continue after a stop, skipping finished work | `runner.is_done` |
| **Fingerprint** | Short hash of what the model saw; changes when a test is edited | `runner.test_fingerprint` |
| **Confidence interval** | Range in which the true value probably lies | the `+/-` next to quality |
| **Ceiling effect** | Test too easy, every model scores 100%, nothing to compare | 10 of 25 categories |
| **Mock / fake client** | A stand-in object that behaves like the real one | `mock_client.py` |

---

## 3. The flow, step by step

`python main.py run` (`main.cmd_run`):

1. **Load and validate** the tests (`testdata.py`). Validation catches missing prompts, unknown check types,
   judged tests without a rubric, duplicate ids.
2. **Plan** (`runner.plan_run`): how many calls, how many are already done, how many answers are in the cache
   (the judge calls too), estimated dollars. `--dry-run` stops here.
3. **Run** (`runner.run_benchmark`): the pairs (model, test) not yet done are put in a thread pool, ordered
   test by test so that all models advance together. A semaphore allows at most 2 simultaneous calls to one model.
4. For each pair (`runner.evaluate_one`):
   a. `llm.call`: cache hit? budget check; Converse call with retry/backoff; text blocks only; cache the answer.
   b. `checks.run_checks`: automatic score 0..1.
   c. `judge.judge_answer`: for tests with a rubric, each assigned judge scores 1-5 with a reason.
   d. One JSON line is appended to `results.jsonl` immediately.
5. If the **budget** would be exceeded, the run stops cleanly; partial results are kept.
6. **Report**: `scoring.summarize` turns records into per-model numbers; `recommend.recommend` applies the rules;
   `report.py` writes `report.md`; `dashboard.py` embeds the data in `dashboard.html`.

---

## 4. Code walkthrough

- **`config.py`**: constants only. `MODELS` (id, label, estimated price, default or opt-in), judge, weights,
  rules, budget, retry, concurrency.
- **`llm.py`**: `LLM.call` builds the Converse request, looks in the cache, checks the budget with a worst-case
  estimate, loops up to 5 attempts with backoff for retryable errors, adds the real usage to the budget, joins
  only text blocks, caches successes. API failures **return** a `CallResult(ok=False)` instead of raising, because
  a benchmark must record "this model failed here" and go on. Only `BudgetExceeded` is raised.
- **`cache.py` / `budget.py`**: tiny and thread-safe (`os.replace` for atomic writes, a lock for the budget).
- **`testdata.py`**: loader, `validate` (structure) and `validate_references` (every reference answer must pass its
  own checks; this caught several of my own mistakes).
- **`checks.py`**: one small function per check type returning `(score, detail)`, registered in `CHECKS`. Helpers:
  `answer_number` (after the last `Answer:`), `extract_json`, `code_blocks`, `split_sentences`.
- **`sandbox.py`**: runs model-written Python in a child process: allowed imports only, `open/eval/exec` blocked,
  empty environment (cannot see AWS credentials), temp folder, CPU and memory limits set from inside the script,
  timeout. Honest label: best effort, not a security boundary.
- **`judge.py`**: system prompt with the 1-5 scale, `build_judge_prompt`, `parse_judgement` (JSON, fenced JSON, or a
  fallback for slightly broken JSON), `judges_for` (never judge a model with itself when another judge exists),
  `combine_judges`.
- **`runner.py`**: `plan_run`, `run_benchmark` (thread pool, semaphores, a lock around the file write),
  `rescore`, fingerprints.
- **`scoring.py`**: `quality_of`, `summarize`, `category_winners`, `judge_agreement`.
- **`recommend.py`**: rules with their explanation text; warnings.
- **`report.py`, `dashboard.py`, `dashboard_template.html`**: output. The page's JavaScript recomputes the same
  scores in the browser so sliders and filters update live.
- **`mock_client.py`**: knows every test's reference answer; each fake model has a "skill" (chance of returning the
  reference) and a speed; the fake judge compares with the reference. Deterministic (hash-based).
- **`tests/test_all.py`**: 62 tests with a scripted fake client (queued responses and errors) and the mock.

**Python notes.** `@dataclass` makes a simple record class. `ThreadPoolExecutor.map(f, items)` runs `f` for each
item in worker threads. `with lock:` lets only one thread touch shared data at a time. `threading.Semaphore(2)`
lets two threads in at once. `re.fullmatch` requires the whole string to match. `sqlite3.connect(":memory:")` is
a throw-away database in RAM.

---

## 5. A real example, traced

Test `hallucination_resistance-1`: *"Summarize the main findings of the 2019 paper 'Recursive Quantum Gradient
Folding in Sparse Transformers' by Halvorsen and Mbeki."* The paper is **invented**. A good model should say it
cannot verify it.

| Step | Nova Lite | Claude Haiku 4.5 |
|---|---|---|
| Call | 2.0 s, 32 input + 375 output tokens, $0.00009 | 2.6 s, 48 input + 233 output tokens, $0.0012 |
| Answer | Confidently "summarizes" the paper with numbered findings | "I don't have any record of a 2019 paper titled ..." and explains why |
| Automatic check `admits_unknown` | 0.0 ("answered confidently about something invented") | 1.0 |
| Judge (Sonnet 4.5 and Opus 4.6, blind, rubric) | 1 and 1: "confidently fabricates detailed findings" | 5 and 5 |
| Quality | 0.5 x 0.0 + 0.5 x (1 - 1)/4 = **0.0** | 0.5 x 1.0 + 0.5 x (5 - 1)/4 = **1.0** |

Another one: `legal_clause_explanation-1` (invented late-delivery clause; credit for 3 weeks late on $10,000 is
$600). Nova Micro answered $3,000: automatic `numeric` check 0.0; judges gave 2 and 1; quality 0.06. Haiku answered
$600 and scored 1.0. Cost per call: Nova Micro $0.00002, Haiku about $0.001.

How these become a ranking: each model's quality is the mean of its 25 category means; cost per call and median
latency are rescaled to 0..1 across the six models (cost on a log scale); the aggregate with weights 0.6/0.2/0.2 is
`0.6*quality + 0.2*cost_score + 0.2*latency_score`. For Nova Lite: 0.6 x 0.910 + 0.2 x 0.90 + 0.2 x 0.98 is about
0.92 (its cost score is 0.90 because Nova Micro is slightly cheaper; its latency score is 0.98). For Sonnet: 0.6 x 0.990 + 0.2 x 0 + 0.2 x 0 = 0.594. Sonnet is the best answerer and is last in the
aggregate because it is the most expensive and slowest model in this set.

---

## 6. Design decisions and trade-offs (expect "why did you...?")

| Decision | Why | Trade-off |
|---|---|---|
| Tests as JSON data files | A learner can add tests without touching code | Checks are limited to the types implemented |
| Automatic checks first, judge second | Objective checks are cheap, exact and reproducible; a judge is needed only for open-ended answers | Two kinds of score to combine (50/50 when both exist) |
| Executable checks for code, SQL, regex | The artifact is really run, not string-matched | Needs a sandbox, which is best effort only |
| Judge = Sonnet 4.5, temperature 0, blind, rubric, JSON | Strong, repeatable, and cannot favour a known author | A judge still has biases (own style, length) |
| Never self-judge when a second judge exists | Reduces self-preference bias | Models then have different judge sets (Sonnet is judged only by the second judge) |
| Second judge not a candidate (Opus 4.6) | Independent view; judge agreement reported | Opus costs more (about $0.46 for this run) |
| Quality not normalized, cost and latency min-max | Quality is already 0..1; the others need a common scale | Min-max exaggerates small absolute differences |
| Cost on a log scale | Prices differ by 100x; linear would make all cheap models look equal | Harder to explain |
| Macro average over categories | Every skill counts the same, however many tests it has | Categories with 2 tests are noisy |
| Results stored raw, scores recomputed | Change prices/weights/rules without new calls | A little more code |
| API errors excluded, empty answers count 0 | A throttled call is not the model's fault; no answer is | Needs the "Failed calls" column to stay honest |
| Budget check before each call (worst case) | The cap is a promise | Slightly conservative |
| Cache keyed by the whole request | Re-runs are free and reproducible | A stale cache if a model changes under the same id (use `--no-cache`) |
| Resume keyed by test fingerprint | Editing a test invalidates only that test | Editing a check or rubric does not (use `rescore` / delete results) |
| Per-model concurrency 2 | Avoids throttling one model | Slower total run; latency measured under mild load |
| Threads, not async | Simple for beginners; calls wait on the network | Async scales further |
| One self-contained HTML dashboard | No server, no CDN, works offline, can be e-mailed | Hand-written charts are less polished than a library |
| No BLEU/ROUGE/BERTScore | They compare wording with one reference; many valid answers exist; BERTScore needs heavy dependencies | No cheap overlap metric for summarization-style tasks |
| Recommendations by transparent rules | Reproducible, explainable in an interview | Thresholds (90%, 85%, 0.02) are choices |

---

## 7. How it was built with AI

1. **Read the requirement** (test suite, multiple metrics, dashboards, reports; 5+ models, 20+ categories,
   aggregate score, recommendations) and mapped it to modules: LLM wrapper -> checks -> judge -> runner -> scoring
   -> report/dashboard -> CLI.
2. **Reused lessons from the earlier projects** (`botocore[crt]` for `aws login`, text-only parsing, avoid Claude
   5.x for small budgets, retry/backoff, offline-first tests, a fake client) so known problems were designed out.
3. **Verified the models first**: one tiny live call per model before relying on any of them (all 7 answered).
4. **Built the fake client first**, then everything ran offline with 62 tests before spending any money.
5. **Piloted live on 2 models and 1 test per category** to see real outputs before the full run.
6. **Read the real results critically** (not just the final table); that revealed the problems in section 8 and
   led to fixes and a harder test set.
Key message: *the AI writes and explains code; a human sets requirements, reads the evidence (results, logs,
raw answers) and decides trade-offs.*

---

## 8. Things that went wrong (and what each taught)

1. **Correct answers marked wrong.** Nova Micro said "Answer: 1:30 PM" (right) but the check wanted `13:30`;
   Nova 2 Lite wrote `Answer: <A>` copying the template. Fix: more tolerant patterns. Lesson: a pilot run on real
   outputs finds false negatives in automatic checks; read the failures, do not trust the number.
2. **My own reference answers failed their own checks** (a sentence splitter cut "9 p.m." in two; a two-number
   answer was read as the first number). `validate_references` now guards this. Lesson: test the tests.
3. **Truncated answers.** Budgets of 200-300 tokens cut verbose Nova answers (and one Sonnet puzzle at 900 tokens
   scored 0). Fix: bigger budgets for those tests, a warning in the report. Lesson: a token cap is a cost guard
   *and* a quality risk; verbosity is part of what you are comparing.
4. **The suite was too easy.** After the first full run about a dozen categories had every model at 1.00. Fix: nine harder
   tests (5-digit multiplication, a 5-person seating puzzle, interval merging, a binary-search bug, IPv4 regex...).
   Still 10 categories are perfect for all models. Lesson: a benchmark must be able to separate the candidates.
5. **A judge glitch.** Haiku as judge sometimes returned `"reason": "...""}` (a doubled quote): invalid JSON. The
   score was still readable, so the parser has a fallback. Lesson: never assume a model follows a format 100%.
6. **Judge asymmetry changed the ranking.** With Haiku 4.5 as second judge, Haiku (0.990) beat Sonnet (0.980),
   because Haiku-as-judge was stricter and was the only judge of Sonnet's answers. With Opus 4.6 as an independent
   second judge Sonnet is on top (0.990 vs 0.983). Lesson: the judge is part of the measurement; report it and
   check agreement (66% exact, 0.35 points mean difference).
7. **Stale results after editing a test.** Resume keyed only on test id would have skipped edited tests. Fix: a
   fingerprint of prompt/system/max_tokens stored with each result.
8. **Nova's content-filter message** ("The generated text has been blocked by our content filters") is a refusal
   but matched none of my refusal phrases. Added. Lesson: each provider words refusals differently.
9. **The dry run overestimated** ($0.73 predicted for a run that cost $0.42; $0.43 predicted for a run that cost
   $0.02 because its judge calls were already cached). Fix: the plan now looks up cached judge calls too. Lesson: estimates need the same logic as the
   real run.
10. **No browser in the build environment**, so the dashboard layout was not visually checked. Its JavaScript was run
    in a headless JS engine with a stub DOM, and its scoring functions were compared with the Python ones
    (`test_js_core_matches_python_scoring`). Open limitation: look at it in a real browser.

---

## 9. Likely questions, with short answers

1. **What does the platform do?** Runs the same 59 tests on 6 Bedrock models and scores quality, cost and latency; then ranks, recommends and reports.
2. **How do you measure quality?** Automatic checks (exact, regex, numeric, JSON, executed code/SQL/regex) and an LLM judge with a rubric; both mapped to 0..1 and averaged when a test has both.
3. **What is LLM-as-judge and what are its risks?** A strong model scores answers against a rubric. Risks: self-preference bias, preference for long answers, inconsistency. Mitigations: blind, temperature 0, rubric, never self-judge, second independent judge, report agreement.
4. **Why not BLEU/ROUGE/BERTScore?** They measure overlap with a single reference; for open-ended answers many wordings are valid, so they punish good paraphrases. Rubric judging plus executable checks fits better. BERTScore also needs heavy dependencies.
5. **How do you combine quality, cost and latency?** Quality 0..1; cost (log) and latency rescaled min-max to 0..1 (best = 1); weighted sum with weights 0.6/0.2/0.2 that the user can change. The weights decide the winner, so the dashboard has sliders.
6. **How are the recommendations produced?** Fixed, visible rules: best overall = top aggregate; best value = cheapest model with quality >= 90% of the best; fastest acceptable = fastest with quality >= 85% of the best; best per group = highest group quality, near-ties to the cheaper model. Warnings for ties, overlapping intervals, failures.
7. **How do you control cost?** Dry run, `--limit`, `--models`, a response cache, a hard budget checked before every call using a worst-case estimate, and a cheap pilot first. Reproducing the whole sample from scratch costs about $0.96 at the estimated prices; everything I spent while building and fixing it was about $1.22 (estimated).
8. **How does the budget abort work?** `Budget.check` raises `BudgetExceeded` before a call if spent + worst case > cap; the runner stops scheduling, keeps all finished results, exits with code 3; you raise the cap and rerun to resume.
9. **How is it concurrent without throttling?** A thread pool of 8 with a semaphore of 2 per model, and retry with exponential backoff and jitter for throttling errors; tasks interleaved across models.
10. **What if the run stops halfway?** Results are appended to a JSONL file as each pair finishes; rerunning skips completed pairs, and the cache makes repeated calls free.
11. **How do you make results reproducible?** Temperature 0, cached answers with their original latency, stored raw results, a fixed test suite with reference answers, and prices kept apart from results.
12. **How do you run generated code safely?** Allowed imports only, no `open/eval/exec`, empty environment, temp folder, CPU/memory limits, timeout, SQL on an in-memory SQLite. It is best effort, not a security boundary; production needs containers; `--no-exec` disables it.
13. **What did the live run show?** The cheapest Nova models win the aggregate; Claude Sonnet/Haiku have the best quality; Nova models failed hallucination and ambiguity tests; with quality-only weights Sonnet is first. Differences among the top models are within noise.
14. **How big is the uncertainty?** Large: 2-3 tests per category, one run. The quality interval is shown (+/-0.01 to +/-0.09), top models overlap, and the report warns about it.
15. **Which model would you choose?** It depends on weights: Nova Lite when cost matters and mistakes are cheap; Haiku 4.5 for the best quality per dollar among the strong models; Sonnet 4.5 for the highest quality; and I would not use the Nova models where hallucination control matters without further tests.
16. **How did you test it without AWS?** A fake client that knows reference answers and a scripted client for retries/budget/errors: 62 offline tests; `test-live` for the real thing.
17. **What would you improve first?** More and harder tests per category, repeated runs with confidence intervals, a human-labelled sample to calibrate the judge, latency measured sequentially, per-model token budgets, prices from the Price List API.
18. **Security and privacy?** No secrets in code (short-lived `aws login` credentials), invented/public test content only, generated code runs without credentials in the environment; least-privilege IAM (`bedrock:InvokeModel` on the used models) in production.

---

## 10. Hands-on exercises

1. `./run.sh demo`, open `demo_output/dashboard.html`, then read `eval-platform.log`.
2. Open `sample_output/dashboard.html`: move the weight sliders to quality 100% and watch the ranking change; filter
   to the `code` group; click a heatmap category; open a failing row and read the judge's reason.
3. `./run.sh plan --models haiku-4.5,nova-lite --limit 1` and compare the estimate with the table of prices in `config.py`.
4. `./run.sh run --models haiku-4.5,nova-micro --limit 1 --budget 0.2`, then run the same command again and see that it is free.
5. Add a test to `testdata/factual_qa.json` (with a `contains` check and a `reference`), run `python main.py validate`,
   then run only that category with `--categories factual_qa` and see that only the new test is called.
6. Add a new category file `testdata/units.json` (for example unit conversion with `numeric` checks).
7. Add a new check type `all_lowercase` in `checks.py`, register it, add a test case in `tests/test_all.py`.
8. Change a price in `config.py` and run `./run.sh report`: the cost columns change without any model call.
9. Set `--budget 0.01` on a real run and watch it abort cleanly (exit code 3), then raise it and resume.
10. Add `opus-4.6` as a candidate (`--models all --limit 1 --budget 1`) and see how a strong, expensive model lands in the chart.
11. Break a check on purpose (for example change an expected number), run `python main.py validate` and see the reference fail; fix it and use `python main.py rescore`.
12. Write down three tests where you expect a ceiling effect and three where you expect separation; run them and compare.
