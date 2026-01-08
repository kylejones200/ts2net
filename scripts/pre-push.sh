#!/bin/bash
# Git pre-push hook: Run tests before allowing push
# This prevents pushing broken code to GitHub

set -e

echo "🔍 Running pre-push checks..."

# Check if we're pushing to main/master (most important)
protected_branches=('main' 'master')
current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

if [[ " ${protected_branches[@]} " =~ " ${current_branch} " ]]; then
    echo "⚠️  Pushing to protected branch: $current_branch"
    echo "Running full test suite..."
    
    # Verify we're in a clean environment that matches CI
    # Check if dev dependencies are installed (especially torch for temporal_cnn tests)
    if ! python -c "import torch" 2>/dev/null; then
        echo "⚠️  PyTorch not found. Installing dev dependencies to match CI..."
        pip install -e .[dev] > /dev/null 2>&1 || {
            echo "❌ Failed to install dev dependencies!"
            echo "Run: pip install -e .[dev]"
            exit 1
        }
    fi
    
    # Run the same tests as CI
    if command -v pytest &> /dev/null; then
        echo "Running pytest..."
        # Disable pytest-xdist to avoid conflicts with joblib parallel workers
        # Skip pynndescent tests (optional dependency) that cause segfaults in some environments
        # Exclude test_multivariate_extended.py which has pynndescent tests
        pytest -q -p no:xdist --ignore=tests/test_multivariate_extended.py || {
            echo ""
            echo "❌ Tests failed! Push aborted."
            echo "Run 'make test' or 'pytest -q' locally to debug."
            exit 1
        }
    else
        echo "⚠️  pytest not found. Install with: pip install pytest"
        echo "Skipping tests (not recommended for protected branches)"
    fi
else
    echo "Pushing to branch: $current_branch (not protected)"
    echo "Skipping tests (use 'make test' manually if needed)"
fi

echo "✅ Pre-push checks passed!"
exit 0



