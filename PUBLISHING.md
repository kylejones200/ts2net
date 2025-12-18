# Publishing Guide

This guide explains how to publish `ts2net` to PyPI and set up documentation on ReadTheDocs.

## PyPI Publishing

### Initial Setup

1. **Create a PyPI account** at https://pypi.org/account/register/

2. **Update project URLs** in `pyproject.toml`:
   ```toml
   [project.urls]
   Homepage = "https://github.com/YOURUSERNAME/ts2net"
   Repository = "https://github.com/YOURUSERNAME/ts2net"
   Issues = "https://github.com/YOURUSERNAME/ts2net/issues"
   Documentation = "https://ts2net.readthedocs.io"
   ```

3. **Set up Trusted Publishing** (recommended, no API tokens needed):
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher:
     - PyPI Project Name: `ts2net`
     - Owner: Your GitHub username
     - Repository name: `ts2net`
     - Workflow name: `publish-pypi.yml`
     - Environment name: (leave blank)

4. **Alternative: Use API Token** (if not using trusted publishing):
   - Generate an API token at https://pypi.org/manage/account/token/
   - Add it to GitHub Secrets as `PYPI_API_TOKEN`
   - Uncomment the `password` line in `.github/workflows/publish-pypi.yml`

### Publishing a Release

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.5.0"
   ```

2. **Create a git tag**:
   ```bash
   git tag v0.5.0
   git push origin v0.5.0
   ```

3. **Create a GitHub Release**:
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Select the tag you just created
   - Add release notes
   - Click "Publish release"

4. The GitHub Action will automatically build wheels for Linux, macOS, and Windows, and publish to PyPI.

### Manual Publishing (for testing)

```bash
# Install build tools
pip install maturin twine

# Build wheels
maturin build --release

# Build source distribution
maturin sdist

# Upload to PyPI
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

## ReadTheDocs Setup

### Initial Setup

1. **Create a ReadTheDocs account** at https://readthedocs.org/accounts/signup/

2. **Import your project**:
   - Go to https://readthedocs.org/dashboard/
   - Click "Import a Project"
   - Connect your GitHub account if needed
   - Select the `ts2net` repository
   - Click "Next"

3. **Configure the project**:
   - Project name: `ts2net`
   - Repository URL: Your GitHub repository URL
   - Default branch: `main` (or `master`)
   - Click "Finish"

4. **Enable webhooks** (should be automatic):
   - ReadTheDocs will automatically set up a webhook to rebuild docs on push
   - Check under Settings → Integrations if needed

5. **Configure advanced settings** (optional):
   - Go to Admin → Advanced Settings
   - Set default version to "latest" or "stable"
   - Enable "Build pull requests" for PR previews
   - Enable "Privacy Level: Public"

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt
pip install -e .

# Build the documentation
cd docs
make html

# Open in browser
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

### Documentation Structure

The documentation uses Sphinx with the following structure:

```
docs/
├── conf.py           # Sphinx configuration
├── index.rst         # Main documentation page
├── api.rst           # API reference
├── usage.rst         # Usage examples
└── requirements.txt  # Documentation dependencies
```

### Updating Documentation

Documentation is automatically rebuilt when you:
- Push to the main branch
- Create a new tag/release
- Open a pull request (if enabled)

## Badges for README

Add these badges to your `README.md`:

```markdown
[![PyPI version](https://badge.fury.io/py/ts2net.svg)](https://badge.fury.io/py/ts2net)
[![Documentation Status](https://readthedocs.org/projects/ts2net/badge/?version=latest)](https://ts2net.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/YOURUSERNAME/ts2net/workflows/Tests/badge.svg)](https://github.com/YOURUSERNAME/ts2net/actions)
[![codecov](https://codecov.io/gh/YOURUSERNAME/ts2net/branch/main/graph/badge.svg)](https://codecov.io/gh/YOURUSERNAME/ts2net)
```

## Checklist Before First Release

- [ ] Update `pyproject.toml` with correct GitHub URLs
- [ ] Add/update `README.md` with installation and usage instructions
- [ ] Add/update `CHANGELOG.md` with version history
- [ ] Ensure all tests pass: `pytest tests/`
- [ ] Set up PyPI account and trusted publishing
- [ ] Set up ReadTheDocs account and import project
- [ ] Create first release on GitHub
- [ ] Verify package appears on PyPI
- [ ] Verify documentation builds on ReadTheDocs
- [ ] Test installation: `pip install ts2net`

## Troubleshooting

### PyPI Publishing Issues

- **Build fails**: Check that Rust toolchain is properly installed in CI
- **Import error**: Ensure `maturin` version is compatible
- **Wheels missing**: Check the build matrix in `.github/workflows/publish-pypi.yml`

### ReadTheDocs Issues

- **Build fails**: Check the build logs on ReadTheDocs
- **Import error**: Ensure all dependencies are in `docs/requirements.txt`
- **Rust extension fails**: The `.readthedocs.yaml` includes Rust installation steps
- **Module not found**: Check that `sys.path` is correctly set in `docs/conf.py`

### Common Solutions

1. **Clear ReadTheDocs build cache**: Admin → Versions → Wipe version
2. **Check build logs**: Click on the build number to see detailed logs
3. **Test locally**: Always test documentation builds locally first
4. **Version conflicts**: Pin dependency versions in requirements files

