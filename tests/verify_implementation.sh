#!/bin/bash
# Verification script for the indices/counts caching implementation

echo "========================================================================"
echo "Verifying Indices/Counts Caching Implementation"
echo "========================================================================"
echo ""

# Check if modified file exists
echo "1. Checking if modified file exists..."
if [ -f "src/models/ShareGNN/layers/inv_based_message_passing.py" ]; then
    echo "   ✓ File exists: src/models/ShareGNN/layers/inv_based_message_passing.py"
else
    echo "   ✗ File not found!"
    exit 1
fi
echo ""

# Check Python syntax
echo "2. Checking Python syntax..."
if python3 -m py_compile src/models/ShareGNN/layers/inv_based_message_passing.py 2>/dev/null; then
    echo "   ✓ Python syntax is valid"
else
    echo "   ✗ Syntax errors detected!"
    exit 1
fi
echo ""

# Check for required methods
echo "3. Checking for required methods..."
if grep -q "def get_cache_path" src/models/ShareGNN/layers/inv_based_message_passing.py; then
    echo "   ✓ get_cache_path() method found"
else
    echo "   ✗ get_cache_path() method not found!"
    exit 1
fi

if grep -q "def _load_cached_indices" src/models/ShareGNN/layers/inv_based_message_passing.py; then
    echo "   ✓ _load_cached_indices() method found"
else
    echo "   ✗ _load_cached_indices() method not found!"
    exit 1
fi

if grep -q "def _save_cached_indices" src/models/ShareGNN/layers/inv_based_message_passing.py; then
    echo "   ✓ _save_cached_indices() method found"
else
    echo "   ✗ _save_cached_indices() method not found!"
    exit 1
fi
echo ""

# Check for bug fix (variable initialization before try-except)
echo "4. Checking for variable scoping bug fix..."
if grep -q "do_invalid_indices_exist = False" src/models/ShareGNN/layers/inv_based_message_passing.py; then
    # Check if it comes before the try-except block
    line_num_init=$(grep -n "do_invalid_indices_exist = False" src/models/ShareGNN/layers/inv_based_message_passing.py | head -1 | cut -d: -f1)
    line_num_try=$(grep -n "try:" src/models/ShareGNN/layers/inv_based_message_passing.py | grep -A5 "_load_cached_indices" | head -1 | cut -d: -f1)

    if [ "$line_num_init" -lt "$line_num_try" ]; then
        echo "   ✓ Variable scoping bug fixed (initialization before try-except)"
    else
        echo "   ⚠ Warning: Variable initialization may be in wrong location"
    fi
else
    echo "   ✗ Variable scoping bug fix not found!"
    exit 1
fi
echo ""

# Check for cache hit/miss messages
echo "5. Checking for cache hit/miss messages..."
if grep -q "Cache hit:" src/models/ShareGNN/layers/inv_based_message_passing.py; then
    echo "   ✓ Cache hit message found"
else
    echo "   ✗ Cache hit message not found!"
    exit 1
fi

if grep -q "Cache miss:" src/models/ShareGNN/layers/inv_based_message_passing.py; then
    echo "   ✓ Cache miss message found"
else
    echo "   ✗ Cache miss message not found!"
    exit 1
fi
echo ""

# Check for required imports (should already exist)
echo "6. Checking for required imports..."
required_imports=("import hashlib" "import json" "from pathlib import Path" "from datetime import datetime")
for import_line in "${required_imports[@]}"; do
    if grep -q "$import_line" src/models/ShareGNN/layers/inv_based_message_passing.py; then
        echo "   ✓ Found: $import_line"
    else
        echo "   ✗ Missing: $import_line"
        exit 1
    fi
done
echo ""

# Check for test files
echo "7. Checking for test files..."
if [ -f "tests/test_cache_methods.py" ]; then
    echo "   ✓ Unit test file exists: tests/test_cache_methods.py"
else
    echo "   ⚠ Unit test file not found (optional)"
fi

if [ -f "test_cache_implementation.py" ]; then
    echo "   ✓ Integration test file exists: test_cache_implementation.py"
else
    echo "   ⚠ Integration test file not found (optional)"
fi
echo ""

# Summary
echo "========================================================================"
echo "Verification Complete!"
echo "========================================================================"
echo ""
echo "✓ All critical checks passed"
echo ""
echo "Next steps:"
echo "  1. Run integration test: python3 test_cache_implementation.py"
echo "  2. Test manually with: cd src && python3 -m examples.basic_example_share_gnn.main"
echo "  3. Check cache directory: results/MUTAG/ShareGNN_indices_cache/"
echo ""
