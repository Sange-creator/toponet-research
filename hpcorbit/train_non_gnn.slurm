#!/usr/bin/env bash
#SBATCH --job-name=toponet-mlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation/%x_%A_%a.out
#SBATCH --error=logs/ablation/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Set PROJECT_ROOT or submit from the project root}}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs/ablation_non_gnn}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/ablation/non_gnn}"
SETUP_SCRIPT="${HPCORBIT_SETUP:-$PROJECT_ROOT/hpcorbit/hpcorbit_env.sh}"

if [[ -f "$SETUP_SCRIPT" ]]; then
    # The site-specific file may load modules and activate a Python environment.
    source "$SETUP_SCRIPT"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if (( ${#CONFIGS[@]} == 0 )); then
    echo "No ablation JSON configurations found in $CONFIG_DIR" >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "Array index $SLURM_ARRAY_TASK_ID is outside the configuration list" >&2
    exit 2
fi

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="$(basename "${CONFIG%.json}")"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -c 'import torch, toponet; print("Python", __import__("sys").version.split()[0], "PyTorch", torch.__version__)'

ARGS=(hpcorbit/train_non_gnn.py --project "$PROJECT_ROOT" --config "$CONFIG" --output "$RUN_DIR")
if [[ "${RESUME:-0}" == "1" ]]; then
    ARGS+=(--resume)
fi
srun "$PYTHON_BIN" "${ARGS[@]}"
