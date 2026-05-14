#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$ROOT/scripts/easy_dataset_workflow.py" "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python "$ROOT/scripts/easy_dataset_workflow.py" "$@"
fi

echo "Python wurde nicht gefunden."
echo "Bitte installiere Python 3 und fuehre danach diese Datei erneut aus."
exit 1
