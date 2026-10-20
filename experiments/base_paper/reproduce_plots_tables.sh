#!/bin/bash
# Regenerate all base_paper figures and LaTeX tables from existing result CSVs
# and trained models. This script does NOT train anything -- run the experiment
# scripts (TUDatasets.sh, ZINC.sh, substructure_counting.sh, synthetic.sh, ...)
# first.
#
# Usage:
#   ./reproduce_plots_tables.sh                 # all steps
#   ./reproduce_plots_tables.sh tables figures  # selected steps
#
# Steps: tables, figures, zinc, substructure
#
# Note: every script skips outputs that already exist. Delete the PDFs/txt files
# you want rebuilt before re-running.

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# ROOT_DIR is the repository root (two levels up from experiments/base_paper)
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Root directory: $ROOT_DIR"

# PYTHONPATH points at the package source so `import simplegnn` resolves
export PYTHONPATH="$ROOT_DIR/src"
echo "PYTHONPATH: $PYTHONPATH"

# Plotting is headless: never pop up (blocking) GUI windows
export MPLBACKEND=Agg

# Figure generation is single-run matplotlib/lualatex work, no config parallelism
export OMP_NUM_THREADS=1

# Activate virtual environment
ENV_DIR="$ROOT_DIR/venv/bin/activate"
echo "Environment directory: $ENV_DIR"
if [ ! -d "$ROOT_DIR/venv" ] || [ ! -f "$ENV_DIR" ]; then
    echo "Error: Virtual environment not found at $ROOT_DIR/venv. Please run ./install.sh first."
    exit 1
fi

# Change to repository root (config and result paths are relative to repo root)
cd "$ROOT_DIR" || { echo "Failed to change directory to root"; exit 1; }
source "$ENV_DIR" || { echo "Failed to activate virtual environment"; exit 1; }
echo "Virtual environment activated"

# The figures are saved through the matplotlib pgf backend, which shells out to
# lualatex; without it every save_latex_figure call fails.
if ! command -v lualatex > /dev/null 2>&1; then
    echo "Warning: lualatex not found in PATH. Figure steps will fail (pgf backend);"
    echo "         the 'tables' step does not need it."
fi

# Steps to run: all of them unless named on the command line
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=(tables figures zinc substructure)
fi

FAILED=()

# run_step <name> <script path> -- keeps going on failure so that a missing
# result set for one step does not block the remaining ones
run_step() {
    local name="$1"
    local script="$2"
    echo ""
    echo "=============================================================="
    echo ">>> $name: python $script"
    echo "=============================================================="
    if python "$script"; then
        echo "<<< $name: done"
    else
        echo "<<< $name: FAILED (see output above; usually missing results/ or models)"
        FAILED+=("$name")
    fi
}

for step in "${STEPS[@]}"; do
    case "$step" in
        # LaTeX tables -> results/base_paper/tables/
        tables)
            run_step "tables" "experiments/base_paper/src/latex.py"
            ;;
        # Paper figures (ablation, network, shared weights, synthetic)
        figures)
            run_step "figures" "experiments/base_paper/src/latex_plots.py"
            ;;
        # ZINC message-passing figure
        zinc)
            run_step "zinc" "experiments/base_paper/src/plot_zinc.py"
            ;;
        # Substructure-counting figures
        substructure)
            run_step "substructure" "experiments/base_paper/regression/substructure_counting/plot_substructure_counting.py"
            ;;
        *)
            echo "Unknown step: $step (valid: tables, figures, zinc, substructure)"
            FAILED+=("$step (unknown)")
            ;;
    esac
done

# Transfer-learning reports are not included here: transfer_report.py needs
# per-experiment --arm/--output arguments, e.g.
#   python experiments/base_paper/src/transfer_report.py \
#       --arm "scratch=results/transfer/scratch" --arm "finetune=results/transfer/finetune" \
#       --output results/transfer/report.md

echo ""
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All steps completed: ${STEPS[*]}"
    echo "Output: results/base_paper/figures/ (all figures, incl. positions/ layout cache)"
    echo "        results/base_paper/tables/  (LaTeX tables)"
    exit 0
else
    echo "Completed with failures in: ${FAILED[*]}"
    exit 1
fi
