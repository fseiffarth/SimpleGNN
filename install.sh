#!/bin/bash

# Exit on error
set -e

# Note: To make this script executable, run:
# chmod +x install.sh

# --- Select a Python interpreter (prefer newest 3.13 -> 3.10) ---
select_python() {
    for v in 3.13 3.12 3.11 3.10; do
        if command -v "python$v" &> /dev/null; then
            echo "python$v"
            return 0
        fi
    done
    return 1
}

# --- Decide accelerator, target venv, and PyTorch wheel index ---
# Priority: NVIDIA CUDA > AMD ROCm (opt-in) > CPU.
# Each accelerator gets its own venv so a CPU env can coexist with a GPU one.
VENV_DIR="venv"
TORCH_SPEC="torch~=2.10.0"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_LABEL="CPU-only"

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Selecting the CUDA 12.6 build of PyTorch..."
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    TORCH_LABEL="CUDA 12.6"
elif lspci 2>/dev/null | grep -iE "VGA|3D|Display" | grep -qi amd; then
    echo "An AMD GPU was detected:"
    lspci | grep -iE "VGA|3D|Display" | grep -i amd | sed 's/^/  /'
    echo ""
    echo "PyTorch ROCm wheels bundle their own ROCm runtime, so a system ROCm"
    echo "install (rocminfo / /opt/rocm) is NOT required."
    echo ""
    REPLY_ROCM="n"
    if [ -t 0 ]; then
        read -rp "Install the ROCm build of PyTorch into a separate 'venv-rocm'? [y/N] " REPLY_ROCM
    else
        echo "Non-interactive shell detected; defaulting to the CPU build."
        echo "Re-run this script interactively (or create 'venv-rocm' manually) to use ROCm."
    fi
    case "$REPLY_ROCM" in
        [yY] | [yY][eE][sS])
            # ROCm wheels lag the CPU/CUDA line: newest published is 2.9.1+rocm6.4.
            VENV_DIR="venv-rocm"
            TORCH_SPEC="torch~=2.9.1"
            TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"
            TORCH_LABEL="ROCm 6.4"
            ;;
        *)
            echo "Skipping ROCm; installing the CPU build into 'venv'."
            ;;
    esac
fi

# --- Create the virtual environment if needed ---
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation..."
else
    echo "Creating a Python virtual environment in '$VENV_DIR'..."
    if PYTHON_CMD=$(select_python); then
        echo "Using $PYTHON_CMD..."
    else
        echo "Error: None of Python 3.13, 3.12, 3.11, or 3.10 is installed."
        echo ""
        echo "Please install Python 3.13, 3.12, 3.11, or 3.10 and try again."
        echo ""
        echo "Installation hints:"
        echo "- Ubuntu/Debian: sudo apt-get update && sudo apt-get install python3.13 python3.13-venv python3.13-dev"
        echo "- macOS: brew install python@3.13"
        echo "- Windows: Download from https://www.python.org/downloads/"
        echo "- Using pyenv: pyenv install 3.13"
        echo ""
        exit 1
    fi
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# --- Activate the virtual environment ---
echo "Activating virtual environment '$VENV_DIR'..."
source "$VENV_DIR/bin/activate"

# --- Install the selected PyTorch build first ---
echo "Installing PyTorch ($TORCH_LABEL): $TORCH_SPEC"
pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

# --- Install dependencies from requirements.txt ---
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Install SimpleGNN package in editable mode ---
echo "Installing SimpleGNN package in editable mode..."
pip install -e .

echo ""
echo "Installation complete! Activate environment with: source $VENV_DIR/bin/activate"
echo "Verify installation: python -c 'import simplegnn; print(simplegnn.__version__)'"
if [ "$VENV_DIR" = "venv-rocm" ]; then
    echo ""
    echo "ROCm note: integrated Radeon GPUs (e.g. 880M/890M, gfx1103/gfx1150) need a"
    echo "gfx-version override at runtime. If torch does not see the GPU, export:"
    echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0"
    echo "Check with: python -c 'import torch; print(torch.cuda.is_available())'"
fi
