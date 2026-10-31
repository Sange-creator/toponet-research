#!/usr/bin/env bash
#SBATCH --job-name=toponet-mlp-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ablation/eval_%x_%j.out
#SBATCH --error=logs/ablation/eval_%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Set PROJECT_ROOT or submit from the project root}}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
SETUP_SCRIPT="${HPCORBIT_SETUP:-$PROJECT_ROOT/hpcorbit/hpcorbit_env.sh}"
if [[ -f "$SETUP_SCRIPT" ]]; then
    source "$SETUP_SCRIPT"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
AREA="${AREA:?Set AREA to heldout or pokhara}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/ablation/evaluation/$AREA}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/ablation/non_gnn}"
mapfile -t RUN_DIRS < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo "No non-GNN training run directories found under $RUN_ROOT" >&2
    exit 2
fi

mkdir -p "$PROJECT_ROOT/logs/ablation"
cd "$PROJECT_ROOT"
srun "$PYTHON_BIN" hpcorbit/evaluate_non_gnn.py \
    --project "$PROJECT_ROOT" \
    --area "$AREA" \
    --output "$OUTPUT" \
    --run-dir "${RUN_DIRS[@]}"
