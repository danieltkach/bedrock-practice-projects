#!/usr/bin/env bash
# One entry point for the whole project.
#   ./run.sh setup      create the virtualenv and install dependencies
#   ./run.sh download   fetch the public Supreme Court PDFs into data/
#   ./run.sh ingest     build both search indexes (calls Bedrock; takes a few minutes)
#   ./run.sh chat       start the assistant
#   ./run.sh all        setup + download + ingest + chat
set -euo pipefail
cd "$(dirname "$0")"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PY=".venv/bin/python"

setup() {
  [ -d .venv ] || python3 -m venv .venv
  "$PY" -m pip install -q -r requirements.txt
  echo "setup complete"
}

check_aws() {
  aws sts get-caller-identity >/dev/null 2>&1 || {
    echo "No valid AWS credentials. Run: aws login --remote --region us-east-1" >&2
    exit 1
  }
}

case "${1:-}" in
  setup)    setup ;;
  download) "$PY" download_opinions.py; "$PY" download_transcripts.py ;;
  ingest)   check_aws
            "$PY" ingest.py opinions data/opinions
            "$PY" ingest.py transcripts data/transcripts ;;
  chat)     check_aws; "$PY" assistant.py ;;
  all)      setup; "$0" download; "$0" ingest; "$0" chat ;;
  *)        sed -n '2,7p' "$0"; exit 1 ;;
esac
