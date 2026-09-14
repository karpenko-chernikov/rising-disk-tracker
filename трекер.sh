#!/usr/bin/env bash
# Запуск из папки tracker/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/исходники${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -x "${ROOT}/../rising-disk-tracker/.venv/bin/python" ]]; then
  PY="${ROOT}/../rising-disk-tracker/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
exec "$PY" -m трекер "$@"
