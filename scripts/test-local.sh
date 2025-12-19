#!/bin/bash
# Local test script that mimics CI environment
# Run this before pushing to catch issues early

set -e

echo "🧪 Running local tests (mimicking CI)..."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root"
    exit 1
fi

# Install dependencies (same as CI)
echo "📦 Installing dependencies..."
pip install -U pip maturin pytest pytest-cov PyYAML || true
pip install -e . || true
pip install numba tslearn pynndescent pyreadr || echo "Some optional deps failed, continuing..."

# Run tests (same as CI)
echo ""
echo "🔍 Running tests..."
PYTHONHASHSEED=0 pytest -q

echo ""
echo "✅ All tests passed! Safe to push."
