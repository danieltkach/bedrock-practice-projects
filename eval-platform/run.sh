#!/usr/bin/env bash
# One entry point for the whole project.
#   ./run.sh setup                   create the virtualenv and install dependencies
#   ./run.sh test                    offline unit tests (no AWS needed)
#   ./run.sh demo                    full pipeline with a fake client (no AWS, free) -> demo_output/
#   ./run.sh plan [options]          dry run: planned calls and estimated cost, nothing is called
#   ./run.sh run [options]           LIVE benchmark on Bedrock (paid, hard budget cap, resumable)
#   ./run.sh report [options]        rebuild report.md + dashboard.html from saved results (free)
#   ./run.sh dashboard               same as report; prints where the dashboard file is
#   ./run.sh test-live               tiny end-to-end check on real Bedrock (a few cents)
# Options after run/plan/report go straight to main.py, e.g.:
#   ./run.sh run --models haiku-4.5,nova-lite --limit 1 --budget 1
set -euo pipefail
cd "$(dirname "$0")"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PY=".venv/bin/python"

setup() {
  [ -d .venv ] || python3 -m venv .venv
  "$PY" -m pip install -q -r requirements.txt
  echo "setup complete"
}

ensure_setup() { [ -x "$PY" ] || setup; }

check_aws() {
  aws sts get-caller-identity >/dev/null 2>&1 || {
    echo "No valid AWS credentials. Run: aws login --remote --region us-east-1" >&2
    exit 1
  }
}

cmd="${1:-}"; shift || true
case "$cmd" in
  setup)     setup ;;
  test)      ensure_setup; "$PY" -m unittest discover -s tests -t . -v ;;
  demo)      ensure_setup
             rm -rf demo_output
             "$PY" main.py run --mock --judge2 haiku-4.5 "$@"
             echo; echo "Open demo_output/dashboard.html in a browser; read demo_output/report.md" ;;
  plan)      ensure_setup; "$PY" main.py run --dry-run "$@" ;;
  run)       ensure_setup; check_aws; "$PY" main.py run "$@" ;;
  report)    ensure_setup; "$PY" main.py report "$@" ;;
  dashboard) ensure_setup; "$PY" main.py dashboard "$@"
             echo "Open results/dashboard.html (or the --out folder you used) in a browser" ;;
  test-live) ensure_setup; check_aws; "$PY" live_test.py ;;
  *)         sed -n '2,13p' "$0"; exit 1 ;;
esac
