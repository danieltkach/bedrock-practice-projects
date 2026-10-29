#!/usr/bin/env bash
# Entry point for all three projects.
#   ./run.sh <project> <command> [args...]
# Projects: legal-assistant | multi-agent | eval-platform
# Example:  ./run.sh multi-agent test
set -euo pipefail
cd "$(dirname "$0")"

project="${1:-}"
case "$project" in
  legal-assistant|multi-agent|eval-platform) ;;
  *) sed -n '2,6p' "$0"; exit 1 ;;
esac
shift

exec "./$project/run.sh" "$@"
