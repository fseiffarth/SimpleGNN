#!/bin/bash
# QM9 graph-regression experiment (ShareGNN multi-head).
# Replaces the broken experiments/base_paper/QM9.sh placeholder — see
# specs/19-qm9-migration.md.
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# ROOT_DIR is the repository root (two levels up from experiments/qm9)
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

# The split file is mandatory at FrameworkMain construction; generate it once.
SPLITS="$ROOT_DIR/src/simplegnn/datasets/splits/fixed/QM9_splits.json"
if [ ! -f "$SPLITS" ]; then
    echo "Split file missing, generating it (downloads QM9 on first run)..."
    python -m simplegnn.utils.qm_splits --dataset QM9
fi

# Run the experiment
python experiments/qm9/main_QM9.py --num_threads "$NUM_THREADS"
