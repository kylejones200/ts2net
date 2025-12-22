#!/usr/bin/env python3
"""
Validation script to check if the project is ready for PyPI release.
Run this before creating a release to ensure everything is configured correctly.
"""

import sys
import logging
import tomli
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(path, description):
    """Check if a required file exists."""
    if path.exists():
        logger.info(f"{description}: {path.name}")
        return True
    else:
        logger.error(f"{description} missing: {path}")
        return False

def check_pyproject_toml(project_root):
    """Validate pyproject.toml configuration."""
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        logger.error("pyproject.toml not found")
        return False
    
    try:
        with open(pyproject_path, "rb") as f:
            config = tomli.load(f)
        
        # Check version
        version = config.get("project", {}).get("version")
        if version:
            logger.info(f"Version: {version}")
        else:
            logger.error("Version not specified in pyproject.toml")
            return False
        
        # Check URLs
        urls = config.get("project", {}).get("urls", {})
        placeholder_found = False
        for key, url in urls.items():
            if "yourusername" in url.lower():
                logger.warning(f"Placeholder URL found in {key}: {url}")
                placeholder_found = True
            else:
                logger.info(f"{key}: {url}")
        
        if placeholder_found:
            logger.warning("Update placeholder URLs in pyproject.toml before releasing")
        
        # Check required fields
        required = ["name", "description", "readme", "requires-python", "license", "authors"]
        project_config = config.get("project", {})
        
        for field in required:
            if field in project_config:
                logger.info(f"{field} specified")
            else:
                logger.error(f"{field} missing from [project] section")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error reading pyproject.toml: {e}")
        return False

def main():
    """Run all validation checks."""
    logger.info("ts2net Release Validation")
    
    project_root = Path(__file__).parent.parent
    all_checks_passed = True
    
    # Check required files
    logger.info("Required Files:")
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
    
    # Check GitHub Actions workflows
    logger.info("GitHub Actions:")
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
        logger.error(".github/workflows directory not found")
        all_checks_passed = False
    
    # Check documentation
    logger.info("Documentation:")
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
        logger.error("docs directory not found")
        all_checks_passed = False
    
    # Check pyproject.toml configuration
    logger.info("Configuration:")
    if not check_pyproject_toml(project_root):
        all_checks_passed = False
    
    if all_checks_passed:
        logger.info("All checks passed! Project is ready for release.")
        logger.info("Next steps:")
        logger.info("1. Update version in pyproject.toml")
        logger.info("2. Update CHANGELOG.md with release notes")
        logger.info("3. Update README.md if needed")
        logger.info("4. Commit changes and create a git tag")
        logger.info("5. Create a GitHub release")
        logger.info("6. The GitHub Action will automatically publish to PyPI")
        return 0
    else:
        logger.error("Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    try:
        import tomli
    except ImportError:
        logger.info("Installing required dependency: tomli")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
        import tomli
    
    sys.exit(main())

