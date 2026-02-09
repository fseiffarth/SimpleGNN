#!/bin/bash
# Run ZINC example with ShareGNN

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

# Change to src directory for execution
cd "$ROOT_DIR/src" || { echo "Failed to change directory to src"; exit 1; }

# Run the example
python -m examples.zinc.main
