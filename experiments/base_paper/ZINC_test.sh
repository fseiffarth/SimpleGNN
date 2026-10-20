#!/bin/bash
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# ROOT_DIR is the repository root (two levels up from experiments/base_paper)
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Root directory: $ROOT_DIR"

# PYTHONPATH points at the package source so `import simplegnn` resolves
export PYTHONPATH="$ROOT_DIR/src"
echo "PYTHONPATH: $PYTHONPATH"

# Number of threads for parallel execution
NUM_THREADS=30

# Avoid OpenMP oversubscription when running configs in parallel
if [ $NUM_THREADS -gt 1 ] || [ $NUM_THREADS -eq -1 ]; then
    export OMP_NUM_THREADS=1
fi

# Activate virtual environment
ENV_DIR="$ROOT_DIR/venv/bin/activate"
echo "Environment directory: $ENV_DIR"
if [ ! -d "$ROOT_DIR/venv" ] || [ ! -f "$ENV_DIR" ]; then
    echo "Error: Virtual environment not found at $ROOT_DIR/venv. Please run ./install.sh first."
    exit 1
fi

# Change to repository root (config paths are relative to repo root)
cd "$ROOT_DIR" || { echo "Failed to change directory to root"; exit 1; }
source "$ENV_DIR" || { echo "Failed to activate virtual environment"; exit 1; }
echo "Virtual environment activated"

# Run the experiment(s)
python experiments/base_paper/regression/ZINC/main_ZINC_test.py --num_threads $NUM_THREADS
