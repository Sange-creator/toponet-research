#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs/ablation_non_gnn}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/ablation/non_gnn}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Configuration directory not found: $CONFIG_DIR" >&2
    exit 2
fi

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if (( ${#CONFIGS[@]} == 0 )); then
    echo "No ablation JSON configurations found in $CONFIG_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$PROJECT_ROOT/logs/ablation"
export PROJECT_ROOT CONFIG_DIR OUTPUT_ROOT
ARRAY_LAST_INDEX=$(( ${#CONFIGS[@]} - 1 ))
cd "$PROJECT_ROOT"
sbatch --array="0-${ARRAY_LAST_INDEX}%${MAX_CONCURRENT}" \
    "$PROJECT_ROOT/hpcorbit/train_non_gnn.slurm"
