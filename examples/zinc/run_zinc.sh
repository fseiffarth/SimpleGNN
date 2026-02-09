#!/bin/bash
# Run ZINC example with ShareGNN
#
# Usage:
#   From repository root:
#     ./examples/zinc/run_zinc.sh
#
#   From zinc directory:
#     cd examples/zinc
#     ./run_zinc.sh
#
#   From anywhere:
#     /path/to/repo/examples/zinc/run_zinc.sh
#
# This script will:
#   1. Set PYTHONPATH to the repository root
#   2. Activate the virtual environment
#   3. Run the ZINC example with ShareGNN
#   4. Execute preprocessing, grid search, and evaluation

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Set PYTHONPATH to the repo root directory (two levels up from examples/zinc)
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Root directory: $ROOT_DIR"
export PYTHONPATH=$ROOT_DIR

# Set number of threads for parallel execution
NUM_THREADS=-1

# If running parallel configs, limit OpenMP threads to prevent oversubscription
if [ $NUM_THREADS -gt 1 ] || [ $NUM_THREADS -eq -1 ]; then
    export OMP_NUM_THREADS=1
fi

# Activate virtual environment
ENV_DIR="$ROOT_DIR/venv/bin/activate"
echo "Environment directory: $ENV_DIR"
if [ ! -f "$ENV_DIR" ]; then
    echo "Error: Virtual environment activation script not found at $ENV_DIR"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$ROOT_DIR/venv" ]; then
    echo "Error: Virtual environment not found at $ROOT_DIR/venv"
    exit 1
fi

# Activate the virtual environment
source "$ENV_DIR" || { echo "Failed to activate virtual environment"; exit 1; }
echo "Virtual environment activated"

# Set path to Python executable from venv
PYTHON_BIN="$ROOT_DIR/venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    # Try python3 if python doesn't exist
    PYTHON_BIN="$ROOT_DIR/venv/bin/python3"
    if [ ! -f "$PYTHON_BIN" ]; then
        echo "Error: Python executable not found in virtual environment"
        exit 1
    fi
fi
echo "Using Python: $PYTHON_BIN"

# Change to src directory for execution
cd "$ROOT_DIR/src" || { echo "Failed to change directory to src"; exit 1; }

# Run the example
"$PYTHON_BIN" -m examples.zinc.main
