# AWS Bedrock Practice: Final Projects

Three projects built on Amazon Bedrock, each complete with tests, documentation and a one-command runner.

| # | Project | What it does | Folder |
|---|---|---|---|
| 1 | **Legal Assistant with RAG** | Answers questions about 250 real U.S. Supreme Court PDFs with exact file/page citations, remembers the conversation, and writes executive summaries | [`legal-assistant/`](legal-assistant/) |
| 2 | **Multi-Agent System** | A coordinator plans a query, routes subtasks to specialist agents (analysis, writing, code) running in parallel, and merges the results, reporting cost and latency | [`multi-agent/`](multi-agent/) |
| 3 | **Evaluation Platform** | Benchmarks 6 models across 25 test categories on quality, cost and latency, then produces a ranking, recommendations, a Markdown report and an interactive HTML dashboard | [`eval-platform/`](eval-platform/) |

Each folder has its own `README.md` (architecture, configuration, limitations, troubleshooting) and
`STUDY_GUIDE.md` (concepts, code walkthrough, design trade-offs, rehearsal Q&A).

## Try it without AWS (nothing to configure)

Projects 2 and 3 include a built-in fake model client, so you can run their tests and a full demo offline and for free:

```bash
./run.sh multi-agent setup
./run.sh multi-agent test        # 19 offline tests
./run.sh multi-agent demo        # whole pipeline against a fake client

./run.sh eval-platform setup
./run.sh eval-platform test      # 62 offline tests
./run.sh eval-platform demo      # writes demo_output/ (report + dashboard)
```

To see real results without spending anything, open the committed sample run of Project 3:
[`eval-platform/sample_output/report.md`](eval-platform/sample_output/report.md) and
`eval-platform/sample_output/dashboard.html` (open it in a browser).

## Run it for real (needs AWS)

Prerequisites:
- Python 3.10+ and AWS CLI v2.32+ (needed for `aws login`)
- An AWS account (a sandbox is fine) with Amazon Bedrock available in `us-east-1`
- First-time Claude use: open any Claude model in the Bedrock console playground once and complete the Anthropic
  use-case form; allow a few minutes for it to take effect

```bash
aws login --remote --region us-east-1     # short-lived credentials, valid up to 12 hours

./run.sh legal-assistant all              # setup + download PDFs + build indexes + chat (several minutes)
./run.sh multi-agent chat                 # after: ./run.sh multi-agent setup
./run.sh eval-platform plan               # dry run: shows number of calls and estimated cost
./run.sh eval-platform run                # live run, with a hard budget cap (default about $3)
```

Run `./run.sh <project>` with no command to see that project's available commands.

## Cost and safety

- Real Bedrock calls cost money. Prices in the code are hand-typed **estimates**; verify them against current AWS pricing.
- Project 1's ingestion makes about 30,000 embedding calls (cents). Project 3 enforces a budget cap and caches responses so re-runs are free.
- All data is public (Supreme Court documents) or invented. No credentials or customer data are stored in this repository.
- Project 1 is a research aid, not legal advice.

## Repository layout

```
run.sh                  entry point: ./run.sh <project> <command>
legal-assistant/        Project 1: RAG over PDFs
multi-agent/            Project 2: coordinator + specialist agents
eval-platform/          Project 3: model benchmarking
```

Large reproducible files (`data/`, `indexes/`, virtual environments, caches) are not committed; the run scripts recreate them.

## Requirements and versions

Each project pins its own dependencies in `requirements.txt` (`boto3`, `botocore[crt]`, and for Project 1 also `pypdf` and `numpy`) and creates its own virtual environment inside its folder.
