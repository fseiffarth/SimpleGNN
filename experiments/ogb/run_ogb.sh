#!/bin/bash
# OGB graph-property-prediction experiments (ShareGNN multi-head).
# Covers the single-task ogbg-mol* datasets; see experiments/ogb/README.md.
#
# Usage:
#   ./experiments/ogb/run_ogb.sh                # runs molhiv
#   ./experiments/ogb/run_ogb.sh molbace        # runs one dataset
#   ./experiments/ogb/run_ogb.sh --all          # runs all six, in size order
#   NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# ROOT_DIR is the repository root (two levels up from experiments/ogb)
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Root directory: $ROOT_DIR"

# PYTHONPATH points at the package source so `import simplegnn` resolves
export PYTHONPATH="$ROOT_DIR/src"
echo "PYTHONPATH: $PYTHONPATH"

# Number of threads for parallel execution
NUM_THREADS=${NUM_THREADS:-8}

# Avoid OpenMP oversubscription when running configs in parallel
if [ "$NUM_THREADS" -gt 1 ] || [ "$NUM_THREADS" -eq -1 ]; then
    export OMP_NUM_THREADS=1
fi

# Activate virtual environment (venv-rocm if present, otherwise venv)
VENV_DIR="$ROOT_DIR/venv-rocm"
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="$ROOT_DIR/venv"
fi
ACTIVATE="$VENV_DIR/bin/activate"
echo "Environment directory: $VENV_DIR"
if [ ! -f "$ACTIVATE" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR. Please run ./install.sh first."
    exit 1
fi

# Change to repository root (config paths are relative to repo root)
cd "$ROOT_DIR" || { echo "Failed to change directory to root"; exit 1; }
source "$ACTIVATE" || { echo "Failed to activate virtual environment"; exit 1; }
echo "Virtual environment activated"

# Smallest first, so a broken setup fails fast rather than after molhiv.
ALL_DATASETS=(molfreesolv molesol molbace molbbbp mollipo molhiv)

if [ "$1" = "--all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("${1:-molhiv}")
fi

# Split files are mandatory at FrameworkMain construction; generate them once.
# The first call downloads the dataset into /tmp, the same root the experiment's
# preprocessing uses, so the download is shared.
SPLIT_DIR="$ROOT_DIR/src/simplegnn/datasets/splits/fixed"
for ds in "${DATASETS[@]}"; do
    SPLITS="$SPLIT_DIR/ogbg-${ds}_splits.json"
    if [ ! -f "$SPLITS" ]; then
        echo "Split file missing for ogbg-${ds}, generating it (downloads the dataset on first run)..."
        python -m simplegnn.utils.ogb_splits --dataset "ogbg-${ds}"
    fi
done

# Run the experiments
for ds in "${DATASETS[@]}"; do
    echo "=== Running ogbg-${ds} ==="
    python experiments/ogb/main_ogb.py --dataset "$ds" --num_threads "$NUM_THREADS"
done
