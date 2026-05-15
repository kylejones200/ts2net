#!/usr/bin/env python3
"""
Verify PyPI trusted publisher configuration.

This script checks that the workflow configuration matches what should be
configured in PyPI's trusted publisher settings.
"""

import os
import sys
from pathlib import Path

def check_workflow_file():
    """Check that the workflow file exists and has correct structure."""
    workflow_path = Path(".github/workflows/publish-pypi.yml")
    
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        return False
    
    content = workflow_path.read_text()
    
    checks = {
        "Has environment: pypi": 'environment: pypi' in content or 'environment: "pypi"' in content,
        "Has id-token permission": 'id-token: write' in content,
        "Uses pypa/gh-action-pypi-publish": 'pypa/gh-action-pypi-publish' in content,
        "Has publish job": 'publish:' in content or 'publish-pypi:' in content,
    }
    
    print("\n📋 Workflow File Checks:")
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False
    
    print(f"\n  Workflow file: {workflow_path}")
    print(f"  Workflow filename for PyPI: publish-pypi.yml")
    
    return all_pass


def check_pyproject():
    """Check pyproject.toml configuration."""
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    content = pyproject_path.read_text()
    
    checks = {
        "Has project name": 'name = "ts2net"' in content or "name = 'ts2net'" in content,
        "Has version": 'version =' in content,
        "Has build-system": '[build-system]' in content,
    }
    
    print("\n📦 pyproject.toml Checks:")
    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False
    
    # Extract package name
    for line in content.split('\n'):
        if line.strip().startswith('name ='):
            name = line.split('=')[1].strip().strip('"').strip("'")
            print(f"\n  Package name: {name}")
            print(f"  PyPI project URL: https://pypi.org/project/{name}/")
            break
    
    return all_pass


def check_git_remote():
    """Check git remote configuration."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        remote = result.stdout.strip()
        
        if 'github.com' in remote:
            parts = remote.replace('git@github.com:', '').replace('https://github.com/', '').replace('.git', '').split('/')
            if len(parts) == 2:
                owner, repo = parts
                print(f"\n🔗 Git Remote:")
                print(f"  Owner: {owner}")
                print(f"  Repository: {repo}")
                print(f"  Full: {owner}/{repo}")
                return True, owner, repo
        
        print(f"\n⚠️  Could not parse git remote: {remote}")
        return False, None, None
    except Exception as e:
        print(f"\n⚠️  Could not check git remote: {e}")
        return False, None, None


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PyPI Trusted Publisher Configuration Verification")
    print("=" * 60)
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    workflow_ok = check_workflow_file()
    pyproject_ok = check_pyproject()
    remote_ok, owner, repo = check_git_remote()
    
    print("\n" + "=" * 60)
    print("PyPI Trusted Publisher Configuration")
    print("=" * 60)
    print("\nConfigure these settings in PyPI:")
    print(f"  Publisher: GitHub")
    if owner and repo:
        print(f"  Owner: {owner}")
        print(f"  Repository name: {repo}")
    print(f"  Workflow filename: publish-pypi.yml")
    print(f"  Environment name: pypi")
    print(f"\n  PyPI URL: https://pypi.org/manage/project/ts2net/settings/publishing/")
    
    print("\n" + "=" * 60)
    if workflow_ok and pyproject_ok:
        print("✅ All checks passed!")
        print("\nNext steps:")
        print("1. Verify trusted publisher in PyPI matches above settings")
        print("2. Ensure GitHub environment 'pypi' exists (if using environment protection)")
        print("3. Create a test tag: git tag v0.7.1-test && git push origin v0.7.1-test")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

