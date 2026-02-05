#!/bin/bash

# get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# print the script directory for debugging
echo "Script directory: $SCRIPT_DIR"

# set PATH to the root directory of the project (go one level up from the script directory)
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Root directory: $ROOT_DIR"
export PYTHONPATH=$ROOT_DIR

# set number of threads variable
NUM_THREADS=30

# if NUM_THREADS > 1, set OMP_NUM_THREADS to 1
if [ $NUM_THREADS -gt 1 ]; then
    export OMP_NUM_THREADS=1
fi


export OMP_NUM_THREADS=1

# activate virtual environment (echo if it fails)
ENV_DIR="$ROOT_DIR"/venv/bin/activate
echo "Environment directory: $ENV_DIR"
if [ ! -f "$ENV_DIR" ]; then
    echo "Error: Virtual environment activation script not found at $ENV_DIR"
    exit 1
fi

# go to root directory and activate the virtual environment
cd "$ROOT_DIR" || { echo "Failed to change directory to root"; exit 1; }
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# run the script
python paper_experiments/regression/substructure_counting/main_substructure_counting.py --num_threads $NUM_THREADS
