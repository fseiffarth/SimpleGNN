#!/bin/bash
# BREC expressiveness benchmark (Wang & Zhang, 2023), RPC evaluation.
# See specs/21-brec-expressiveness-benchmark.md and experiments/brec/README.md.
#
# Environment variables:
#   MODEL=sharegnn|gin|smoke   which config to run (default: sharegnn)
#   NUM_THREADS=<n>            preprocessing threads (default: 1, see below)
#   PARTS=<a,b>                restrict to categories, e.g. "Basic,Regular"
#   PAIRS=<n>                  only the first n pairs per category
#   EPOCHS=<n>                 epochs per pair (default: the protocol's 50)
#   EPSILON=<x>                covariance ridge (default: 0 = reference)
#
# A default run is 400 pairs x 50 epochs of siamese training and takes hours;
# `MODEL=smoke PARTS=Basic PAIRS=2 EPOCHS=3` is the quick check.
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# ROOT_DIR is the repository root (two levels up from experiments/brec)
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Root directory: $ROOT_DIR"

# PYTHONPATH points at the package source so `import simplegnn` resolves
export PYTHONPATH="$ROOT_DIR/src"
echo "PYTHONPATH: $PYTHONPATH"

# `preprocessing()` parallelizes across DATASETS, and these configs each declare
# exactly one — so >1 buys nothing here and only hits the documented pitfall
# (with num_threads>1 preprocessing runs in a joblib subprocess and in-memory
# splits do not propagate back; see the Known Limitations section of README.md).
# Label and property generation parallelizes internally regardless.
NUM_THREADS=${NUM_THREADS:-1}
MODEL=${MODEL:-sharegnn}

# Avoid OpenMP oversubscription when preprocessing in parallel
if [ "$NUM_THREADS" -gt 1 ] || [ "$NUM_THREADS" -eq -1 ]; then
    export OMP_NUM_THREADS=1
fi

case "$MODEL" in
    sharegnn) CONFIG="experiments/brec/configs/main_config_brec.yml";       DATASET="BREC" ;;
    gin)      CONFIG="experiments/brec/configs/main_config_brec_gin.yml";   DATASET="BREC" ;;
    smoke)    CONFIG="experiments/brec/configs/main_config_brec_smoke.yml"; DATASET="BREC-r4" ;;
    *) echo "Error: MODEL must be one of sharegnn, gin, smoke (got '$MODEL')"; exit 1 ;;
esac
echo "Model: $MODEL (config $CONFIG, dataset $DATASET)"

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
SPLITS="$ROOT_DIR/src/simplegnn/datasets/splits/fixed/${DATASET}_splits.json"
if [ ! -f "$SPLITS" ]; then
    echo "Split file missing, generating it..."
    python -m simplegnn.utils.brec_splits --dataset "$DATASET"
fi

ARGS=(--dataset "$DATASET" --config "$CONFIG" --num_threads "$NUM_THREADS")
[ -n "$PARTS" ]   && ARGS+=(--parts "$PARTS")
[ -n "$PAIRS" ]   && ARGS+=(--pairs "$PAIRS")
[ -n "$EPOCHS" ]  && ARGS+=(--epochs "$EPOCHS")
[ -n "$EPSILON" ] && ARGS+=(--epsilon_matrix "$EPSILON")

# Run the evaluation
python experiments/brec/main_brec.py "${ARGS[@]}"
