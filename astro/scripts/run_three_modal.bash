#!/usr/bin/env bash
# Historical name; launches the selected N-modality YAML without touching other jobs.
# Usage: bash astro/scripts/run_three_modal.bash GPU_INDEX [ECHO_FACTOR] [--config-file YAML]
set -euo pipefail

usage() {
    echo "Usage: ${BASH_SOURCE[0]} GPU_INDEX [ECHO_FACTOR: positive integer] [--config-file YAML]" >&2
    exit 2
}
[[ $# -ge 1 && ${1:-} =~ ^[0-9]+$ ]] || usage
GPU=$1
shift
ECHO_ARGS=()
ECHO_FACTOR=config
if [[ $# -gt 0 && $1 != --config-file ]]; then
    [[ $1 =~ ^0*[1-9][0-9]*$ ]] || usage
    ECHO_FACTOR=$1
    ECHO_ARGS=(--echo-factor "$1")
    shift
fi
CONFIG_FILE=
if [[ $# -gt 0 ]]; then
    [[ $# -eq 2 && $1 == --config-file && -n $2 ]] || usage
    CONFIG_FILE=$2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
LSDB_SRC="$WORKSPACE/lsdb-pr-1584/src"
PYTHON="$REPO_ROOT/.venv-astropt3/bin/python"
CONFIG_FILE=${CONFIG_FILE:-"$REPO_ROOT/astro/configs/nanotron/astropt3-70m-four-modal.yaml"}
if [[ ! -x "$PYTHON" || ! -f "$LSDB_SRC/lsdb/streams/catalog_streams.py" || ! -f "$CONFIG_FILE" ]]; then
    echo "Requires $PYTHON, the LSDB checkout at $LSDB_SRC, and config $CONFIG_FILE" >&2
    exit 1
fi
CONFIG_FILE="$(cd -- "$(dirname -- "$CONFIG_FILE")" && pwd)/$(basename -- "$CONFIG_FILE")"
CONFIG_NAME=$(basename -- "$CONFIG_FILE" .yaml)
OUTPUT="$WORKSPACE/astroPTv3_telemetry/${CONFIG_NAME}-echo${ECHO_FACTOR}-$(date -u +%Y%m%dT%H%M%SZ)-$$"

cd -- "$REPO_ROOT"
printf 'GPU: %s\nOutput: %s\n' "$GPU" "$OUTPUT"
exec env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u NANOTRON_BENCHMARK \
    CUDA_VISIBLE_DEVICES="$GPU" \
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
        --config-file "$CONFIG_FILE" \
        --output "$OUTPUT" \
        "${ECHO_ARGS[@]}"
