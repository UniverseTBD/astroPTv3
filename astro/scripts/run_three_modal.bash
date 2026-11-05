#!/usr/bin/env bash
# Usage: bash astro/scripts/run_three_modal.bash GPU_INDEX [ECHO_FACTOR]
# Starts a fresh four-survey 30k-step run; existing jobs stay untouched.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! ${1:-} =~ ^[0-9]+$ || ! ${2-1} =~ ^0*[1-9][0-9]*$ ]]; then
    echo "Usage: bash ${BASH_SOURCE[0]} GPU_INDEX [ECHO_FACTOR: positive integer]" >&2
    exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
LSDB_SRC="$WORKSPACE/lsdb-pr-1584/src"
PYTHON="$REPO_ROOT/.venv-astropt3/bin/python"
OUTPUT="$WORKSPACE/astroPTv3_telemetry/four-modal-30k-echo${2-1}-$(date -u +%Y%m%dT%H%M%SZ)-$$"

if [[ ! -x "$PYTHON" || ! -f "$LSDB_SRC/lsdb/streams/catalog_streams.py" ]]; then
    echo "Requires $PYTHON and the LSDB checkout at $LSDB_SRC" >&2
    exit 1
fi

cd -- "$REPO_ROOT"
printf 'GPU: %s\nOutput: %s\n' "$1" "$OUTPUT"
exec env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u NANOTRON_BENCHMARK \
    CUDA_VISIBLE_DEVICES="$1" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONPATH="$LSDB_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m torch.distributed.run \
        --nproc_per_node=1 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=127.0.0.1:0 \
        --max-restarts=0 \
        astro/scripts/train_three_catalogs.py \
        --output "$OUTPUT" \
        --echo-factor "${2-1}"
