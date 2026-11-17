#!/bin/bash
# Multi-node N-modal training from an AstroPT3 YAML recipe.
#
# Usage, from the REPO ROOT (paths in the configs are cwd-relative):
#
#   sbatch --nodes=4  astro/scripts/launch_slurm.sbatch astro/configs/nanotron/astropt3-70m.yaml
#   sbatch --nodes=4  astro/scripts/launch_slurm.sbatch astro/configs/nanotron/astropt3-160m.yaml
#   sbatch --nodes=8  astro/scripts/launch_slurm.sbatch astro/configs/nanotron/astropt3-1b.yaml
#
# Node counts per size (recipe table in EXPERIMENTS.md Part I / docs/training.md, 8 GPUs/node):
#   70M/160M/410M: 4   |   1B/1.4B/2.8B: 8   |   6.9B: 16   |   12B: 32
# The config's parallelism must satisfy dp*tp == nodes*8; to run on fewer
# GPUs scale `parallelism.dp` down and `batch_accumulation_per_replica` up.
#
# PLAYBOOK RULE — dry-run before committing node-hours: submit the same
# config with `--export=ALL,DRY_RUN_STEPS=100` first and check
# tokens_per_sec_per_gpu, model_tflops_per_gpu (target ~30-45% MFU) and the
# peak-memory lines; profile via the yaml's `profiler` section if needed.
#
# Environment: $ASTROPT3_ENV must point at the GPU venv (flash-attn +
# editable nanotron/ + editable astro/ + torchdata; recipe in
# docs/training.md §1). Defaults to ../astroPTv3_gpuenv beside the repo.
#
#SBATCH --job-name=astropt3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

CONFIG=${1:?usage: sbatch launch_slurm.sbatch <astro/configs/nanotron/*.yaml>}
ASTROPT3_ENV=${ASTROPT3_ENV:-$SLURM_SUBMIT_DIR/../astroPTv3_gpuenv}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}

source "$ASTROPT3_ENV/bin/activate"

# rendezvous on the first allocated node; port derived from the job id so
# concurrent jobs on a shared node never collide
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

export CUDA_DEVICE_MAX_CONNECTIONS=1   # required by nanotron's comm overlap
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE  # catalog streams require Hub access
export WANDB_MODE=${WANDB_MODE:-online}
export ASTROPT3_TELEMETRY_DIR=${ASTROPT3_TELEMETRY_DIR:-../astroPTv3_telemetry/${SLURM_JOB_ID:-$$}}

# optional dry run: cap train_steps without editing the config
if [[ -n "${DRY_RUN_STEPS:-}" ]]; then
    DRY_CONFIG=$(mktemp --suffix=.yaml)
    python - "$CONFIG" "$DRY_CONFIG" "$DRY_RUN_STEPS" <<'EOF'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
config["tokens"]["train_steps"] = int(sys.argv[3])
config["checkpoints"]["checkpoints_path"] += "-dryrun"
config["general"]["run"] += "-dryrun"
yaml.safe_dump(config, open(sys.argv[2], "w"))
EOF
    CONFIG=$DRY_CONFIG
    echo "DRY RUN: $DRY_RUN_STEPS steps, checkpoints redirected to *-dryrun"
fi

echo "config=$CONFIG nodes=$SLURM_NNODES gpus/node=$GPUS_PER_NODE rdzv=$MASTER_ADDR:$MASTER_PORT"

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id="$SLURM_JOB_ID" \
    --max-restarts=0 \
    nanotron/run_train.py --config-file "$CONFIG"
