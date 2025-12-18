#!/usr/bin/env python3
"""
Validation script to check if the project is ready for PyPI release.
Run this before creating a release to ensure everything is configured correctly.
"""

import sys
import tomli
from pathlib import Path

def check_file_exists(path, description):
    """Check if a required file exists."""
    if path.exists():
        print(f"✓ {description}: {path.name}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False

def check_pyproject_toml(project_root):
    """Validate pyproject.toml configuration."""
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("✗ pyproject.toml not found")
        return False
    
    try:
        with open(pyproject_path, "rb") as f:
            config = tomli.load(f)
        
        # Check version
        version = config.get("project", {}).get("version")
        if version:
            print(f"✓ Version: {version}")
        else:
            print("✗ Version not specified in pyproject.toml")
            return False
        
        # Check URLs
        urls = config.get("project", {}).get("urls", {})
        placeholder_found = False
        for key, url in urls.items():
            if "yourusername" in url.lower():
                print(f"⚠ Placeholder URL found in {key}: {url}")
                placeholder_found = True
            else:
                print(f"✓ {key}: {url}")
        
        if placeholder_found:
            print("⚠ Warning: Update placeholder URLs in pyproject.toml before releasing")
        
        # Check required fields
        required = ["name", "description", "readme", "requires-python", "license", "authors"]
        project_config = config.get("project", {})
        
        for field in required:
            if field in project_config:
                print(f"✓ {field} specified")
            else:
                print(f"✗ {field} missing from [project] section")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading pyproject.toml: {e}")
        return False

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("ts2net Release Validation")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent.parent
    all_checks_passed = True
    
    # Check required files
    print("Required Files:")
    print("-" * 60)
    required_files = {
        "pyproject.toml": "Build configuration",
        "README.md": "Project documentation",
        "LICENSE": "License file",
        "CHANGELOG.md": "Changelog",
        "MANIFEST.in": "Source distribution manifest",
        ".readthedocs.yaml": "ReadTheDocs configuration",
    }
    
    for filename, description in required_files.items():
        if not check_file_exists(project_root / filename, description):
            all_checks_passed = False
    
    print()
    
    # Check GitHub Actions workflows
    print("GitHub Actions:")
    print("-" * 60)
    workflows_dir = project_root / ".github" / "workflows"
    if workflows_dir.exists():
        workflows = {
            "publish-pypi.yml": "PyPI publishing workflow",
            "tests.yml": "Test workflow",
        }
        for filename, description in workflows.items():
            if not check_file_exists(workflows_dir / filename, description):
                all_checks_passed = False
    else:
        print("✗ .github/workflows directory not found")
        all_checks_passed = False
    
    print()
    
    # Check documentation
    print("Documentation:")
    print("-" * 60)
    docs_dir = project_root / "docs"
    if docs_dir.exists():
        doc_files = {
            "conf.py": "Sphinx configuration",
            "index.rst": "Documentation index",
            "requirements.txt": "Documentation dependencies",
        }
        for filename, description in doc_files.items():
            if not check_file_exists(docs_dir / filename, description):
                all_checks_passed = False
    else:
        print("✗ docs directory not found")
        all_checks_passed = False
    
    print()
    
    # Check pyproject.toml configuration
    print("Configuration:")
    print("-" * 60)
    if not check_pyproject_toml(project_root):
        all_checks_passed = False
    
    print()
    print("=" * 60)
    
    if all_checks_passed:
        print("✓ All checks passed! Project is ready for release.")
        print()
        print("Next steps:")
        print("1. Update version in pyproject.toml")
        print("2. Update CHANGELOG.md with release notes")
        print("3. Update README.md if needed")
        print("4. Commit changes and create a git tag:")
        print("   git tag v0.x.x")
        print("   git push origin v0.x.x")
        print("5. Create a GitHub release")
        print("6. The GitHub Action will automatically publish to PyPI")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    try:
        import tomli
    except ImportError:
        print("Installing required dependency: tomli")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
        import tomli
    
    sys.exit(main())

