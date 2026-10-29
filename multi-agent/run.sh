#!/usr/bin/env bash
# One entry point for the whole project.
#   ./run.sh setup              create the virtualenv and install dependencies
#   ./run.sh test               offline unit tests (no AWS needed)
#   ./run.sh demo               full pipeline with a fake client (no AWS needed)
#   ./run.sh chat               interactive assistant on real Bedrock
#   ./run.sh ask "question"     single-shot question on real Bedrock
#   ./run.sh test-live          end-to-end check on real Bedrock (needs credentials)
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
             "$PY" main.py --mock "Hello!"
             "$PY" main.py --mock "Compare SQS and Kinesis for 5,000 events per second, write a Python consumer, and draft an email to my team." ;;
  chat)      ensure_setup; check_aws; "$PY" main.py "$@" ;;
  ask)       ensure_setup; check_aws; "$PY" main.py "$@" ;;
  test-live) ensure_setup; check_aws; "$PY" live_test.py ;;
  *)         sed -n '2,8p' "$0"; exit 1 ;;
esac
