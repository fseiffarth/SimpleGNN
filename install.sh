#!/bin/bash

# Exit on error
set -e

# Note: To make this script executable, run:
# chmod +x install.sh

# Check if virtual environment already exists and is properly set up
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists. Skipping installation..."
else
    echo "Creating a Python virtual environment..."

    # Check if Python 3.13 is available
    if command -v python3.13 &> /dev/null; then
        echo "Python 3.13 found. Using Python 3.13..."
        PYTHON_CMD="python3.13"
    # Check if Python 3.12 is available
    elif command -v python3.12 &> /dev/null; then
        echo "Python 3.12 found. Using Python 3.12..."
        PYTHON_CMD="python3.12"
    # Check if Python 3.11 is available
    elif command -v python3.11 &> /dev/null; then
        echo "Python 3.11 found. Using Python 3.11..."
        PYTHON_CMD="python3.11"
    # Check if Python 3.10 is available
    elif command -v python3.10 &> /dev/null; then
        echo "Python 3.10 found. Using Python 3.10..."
        PYTHON_CMD="python3.10"
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

    # Create a virtual environment
    $PYTHON_CMD -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate


# First install the cpu or cuda version of torch
echo "Installing PyTorch 2.10.0..."
# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. Installing PyTorch with CUDA 12.6 support..."
    pip install torch~=2.10.0 --index-url https://download.pytorch.org/whl/cu126
else
    echo "CUDA is not available. Installing CPU-only version of PyTorch..."
    pip install torch~=2.10.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install SimpleGNN package in editable mode
echo "Installing SimpleGNN package in editable mode..."
pip install -e .

echo ""
echo "Installation complete! Activate environment with: source venv/bin/activate"
echo "Verify installation: python -c 'import simplegnn; print(simplegnn.__version__)'"
