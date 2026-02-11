#!/usr/bin/env python
"""Setup script for SimpleGNN with PyTorch installation validation."""
import sys
import os

# Check Python version
if sys.version_info < (3, 10):
    sys.stderr.write("ERROR: SimpleGNN requires Python 3.10 or later.\n")
    sys.exit(1)

# Only check for PyTorch during actual install, not during build phase
# Skip check if we're just getting build requirements
if 'egg_info' not in sys.argv and 'dist_info' not in sys.argv:
    try:
        import torch
        print(f"✓ Found PyTorch {torch.__version__}")
        major, minor = map(int, torch.__version__.split('.')[:2])
        if major < 2 or (major == 2 and minor < 10):
            print(f"⚠ WARNING: SimpleGNN requires PyTorch 2.10.0+, found {torch.__version__}")
    except ImportError:
        print(
            "\n" + "="*70 + "\n"
            "⚠ WARNING: PyTorch is not installed!\n\n"
            "SimpleGNN requires PyTorch 2.10.0+. Install it with:\n"
            "  CUDA 12.6: pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cu126\n"
            "  CPU-only:  pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cpu\n"
            + "="*70 + "\n"
        )
        # Only fail if explicitly requested via environment variable
        if os.environ.get('SIMPLEGNN_STRICT_TORCH_CHECK') == '1':
            sys.exit(1)

from setuptools import setup
setup()  # All config in pyproject.toml
