#!/bin/bash
cd "$(dirname "$0")"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$(pwd)/automationproject-486823-705aa82eaa96.json}"
# GCS bucket for Veo extend (and optional generation) output
export VEO_OUTPUT_GCS_URI="${VEO_OUTPUT_GCS_URI:-gs://veo3downloadcvgenix/veo-output/}"

# Prefer Python 3.10+ (e.g. from Homebrew: brew install python@3.12)
for py in python3.12 python3.11 python3.10; do
  if command -v "$py" &>/dev/null && "$py" -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$py"
    break
  fi
done
PY="${PY:-python3}"
if ! "$PY" -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Warning: Python 3.10+ not found. Install with: brew install python@3.12" >&2
fi

# Use or create .venv with the chosen interpreter
VENV=".venv"
if [[ ! -d "$VENV" ]]; then
  echo "Creating venv with $PY..."
  "$PY" -m venv "$VENV"
fi
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
exec python -m uvicorn app:app --reload
