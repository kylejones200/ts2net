#!/bin/bash
# Test PyPI publishing workflow
# This creates a test tag to trigger the workflow

set -e

echo "🧪 Testing PyPI Publishing Workflow"
echo "===================================="
echo ""

# Get current version
VERSION=$(grep '^version =' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "Current version in pyproject.toml: $VERSION"
echo ""

# Create test tag
TEST_TAG="v${VERSION}-test-$(date +%Y%m%d-%H%M%S)"
echo "Creating test tag: $TEST_TAG"
git tag "$TEST_TAG"
echo "✅ Tag created locally"
echo ""

echo "Next steps:"
echo "1. Push the tag: git push origin $TEST_TAG"
echo "2. Watch the workflow: https://github.com/kylejones200/ts2net/actions"
echo "3. Check PyPI after workflow completes: https://pypi.org/project/ts2net/"
echo ""
echo "To delete the test tag after testing:"
echo "  git tag -d $TEST_TAG"
echo "  git push origin :refs/tags/$TEST_TAG"

