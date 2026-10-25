#!/usr/bin/env bash
# M-L2 face reproduction — no new architecture
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi
echo "[1/4] eqmem_hard_v2 (hard → SDA-Recency)"
"$PY" research/eqmem_hard_v2/run_v2.py
echo "[2/4] recency_adverse_v2 (adverse → Expanded)"
"$PY" research/recency_adverse_v2/run_adverse_v2.py
echo "[3/4] hybrid_rf (includes iid suite → Expanded)"
"$PY" research/hybrid_rf/run_hybrid.py
echo "[4/4] rebuild face tables"
"$PY" research/face_repro_ml2/build_tables.py
echo "Done. See research/face_repro_ml2/README.md"
