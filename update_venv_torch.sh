#!/bin/bash
# Reinstall PyTorch in an existing venv with a CUDA build that matches the GPU.
# Defaults to the CUDA 12.8 build, which supports both older GPUs (e.g. RTX 2080) and
# Blackwell (e.g. RTX PRO 6000; cu126 wheels only support up to sm_90).
#
# Usage: [CUDA_TAG=cu126] ./update_venv_torch.sh [VENV_DIR]   (default: venv)

set -e

VENV_DIR="${1:-venv}"
TORCH_SPEC="torch~=2.10.0"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Error: '$VENV_DIR' is not a virtual environment (run from the repo root or pass its path)."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found; this script is for NVIDIA GPUs."
    exit 1
fi

echo "GPUs and compute capabilities:"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader

# CUDA 12.8 wheels cover Turing through Blackwell (sm_75 - sm_120), so one venv works on
# mixed machines. Override with CUDA_TAG=cu126 (or another tag) if needed.
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
LABEL="$CUDA_TAG"

source "$VENV_DIR/bin/activate"
echo "Installing PyTorch ($LABEL) into $VENV_DIR..."
pip uninstall -y torch
pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
# Make sure dependent packages still match the new torch
pip install -r requirements.txt
pip install -e .

echo ""
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.ones(2, 2, device="cuda")
    print("test matmul OK:", (x @ x).sum().item() == 8.0)
EOF
