# Setup Checklist for PyPI and ReadTheDocs

Use this checklist to set up your project for automated publishing.

## Pre-Release Setup

### 1. Update Project Metadata

- [ ] Update GitHub URLs in `pyproject.toml` (replace "yourusername" with your actual username):
  ```toml
  [project.urls]
  Homepage = "https://github.com/YOUR_USERNAME/ts2net"
  Repository = "https://github.com/YOUR_USERNAME/ts2net"
  Issues = "https://github.com/YOUR_USERNAME/ts2net/issues"
  Documentation = "https://ts2net.readthedocs.io"
  ```

- [ ] Update badge URLs in `README.md` (replace "yourusername")

- [ ] Update changelog URLs in `CHANGELOG.md` (replace "yourusername")

### 2. Validate Configuration

Run the validation script:
```bash
python scripts/validate_release.py
```

This will check:
- Required files exist
- GitHub Actions workflows are configured
- Documentation files are present
- pyproject.toml is properly configured

### 3. Test Local Build

Test that the package builds correctly:

```bash
# Install build tools
pip install maturin

# Build the package
maturin build --release

# Check the dist/ folder for wheel files
ls -lh dist/
```

### 4. Test Documentation Build

Build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Open `docs/_build/html/index.html` in your browser to verify.

## PyPI Setup

### Option A: Trusted Publishing (Recommended)

1. [ ] Create a PyPI account at https://pypi.org/account/register/

2. [ ] Set up trusted publishing:
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in:
     - **PyPI Project Name**: `ts2net`
     - **Owner**: Your GitHub username
     - **Repository name**: `ts2net`
     - **Workflow name**: `publish-pypi.yml`
     - **Environment name**: (leave blank)

3. [ ] Push your code to GitHub

4. [ ] Create a release (see "Creating a Release" below)

### Option B: API Token (Alternative)

1. [ ] Create a PyPI account at https://pypi.org/account/register/

2. [ ] Generate an API token:
   - Go to https://pypi.org/manage/account/token/
   - Create a token with scope "Entire account" or specific to `ts2net`

3. [ ] Add token to GitHub Secrets:
   - Go to your repository on GitHub
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI token

4. [ ] Uncomment the password line in `.github/workflows/publish-pypi.yml`:
   ```yaml
   password: ${{ secrets.PYPI_API_TOKEN }}
   ```

## ReadTheDocs Setup

1. [ ] Create a ReadTheDocs account at https://readthedocs.org/accounts/signup/

2. [ ] Import your project:
   - Go to https://readthedocs.org/dashboard/
   - Click "Import a Project"
   - Connect GitHub if needed
   - Select `ts2net`
   - Click "Next"

3. [ ] Configure project settings:
   - **Name**: ts2net
   - **Repository URL**: Your GitHub repo URL
   - **Default branch**: main (or master)
   - Click "Finish"

4. [ ] Enable webhooks (usually automatic):
   - Check under Admin → Integrations
   - Should see a GitHub webhook

5. [ ] Configure advanced settings (optional):
   - Admin → Advanced Settings
   - Set default version to "latest" or "stable"
   - Enable "Build pull requests" for PR previews

## Creating a Release

### 1. Prepare the Release

- [ ] Update version in `pyproject.toml`:
  ```toml
  version = "0.5.0"  # or your version
  ```

- [ ] Update `CHANGELOG.md`:
  - Move items from "Unreleased" to a new version section
  - Add date to version header
  - Update comparison links at bottom

- [ ] Update version in `ts2net/__init__.py` if you have a `__version__` variable

- [ ] Test everything works:
  ```bash
  pytest tests/
  python scripts/validate_release.py
  ```

### 2. Commit and Tag

```bash
# Commit version changes
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.5.0"

# Create and push tag
git tag v0.5.0
git push origin main
git push origin v0.5.0
```

### 3. Create GitHub Release

1. [ ] Go to your repository on GitHub
2. [ ] Click "Releases" → "Create a new release"
3. [ ] Click "Choose a tag" → Select `v0.5.0`
4. [ ] Release title: `v0.5.0` (or add descriptive title)
5. [ ] Copy relevant sections from CHANGELOG.md into description
6. [ ] Click "Publish release"

### 4. Monitor Automated Workflows

- [ ] Check GitHub Actions:
  - Go to Actions tab in your repository
  - Watch the "Publish to PyPI" workflow
  - Should build wheels for Linux, macOS, Windows
  - Should publish to PyPI automatically

- [ ] Check ReadTheDocs:
  - Go to https://readthedocs.org/projects/ts2net/builds/
  - Watch the build progress
  - Check for any errors

### 5. Verify Release

- [ ] Check PyPI: https://pypi.org/project/ts2net/
  - Verify version is published
  - Check that description renders correctly
  - Verify badges work

- [ ] Test installation:
  ```bash
  # In a fresh virtual environment
  pip install ts2net
  python -c "import ts2net; print(ts2net.__version__)"
  ```

- [ ] Check documentation: https://ts2net.readthedocs.io/
  - Verify it built successfully
  - Check that API docs are complete
  - Test navigation

## Post-Release

- [ ] Announce the release (Twitter, mailing lists, etc.)
- [ ] Update any dependent projects
- [ ] Start next version in CHANGELOG.md:
  ```markdown
  ## [Unreleased]

  ### Added
  ### Changed
  ### Deprecated
  ### Removed
  ### Fixed
  ### Security
  ```

## Troubleshooting

### PyPI Publishing Fails

**Problem**: Build fails with Rust errors
- **Solution**: Check that Rust toolchain is properly installed in CI
- Check the Actions logs for specific error messages

**Problem**: Wheels not building for all platforms
- **Solution**: Check the matrix in `.github/workflows/publish-pypi.yml`
- Ensure all OS types are included

**Problem**: "Project name already exists"
- **Solution**: The package name is taken. Change name in `pyproject.toml`

### ReadTheDocs Build Fails

**Problem**: Import errors during build
- **Solution**: Ensure all dependencies are in `docs/requirements.txt`
- Check that Rust build succeeds (see `.readthedocs.yaml`)

**Problem**: Rust compilation fails
- **Solution**: ReadTheDocs has limited resources. May need to simplify build
- Check build logs for specific errors

**Problem**: Module not found
- **Solution**: Verify `sys.path` is set correctly in `docs/conf.py`

### Getting Help

- **PyPI**: https://pypi.org/help/
- **ReadTheDocs**: https://docs.readthedocs.io/
- **GitHub Actions**: https://docs.github.com/en/actions
- **Maturin**: https://www.maturin.rs/

For project-specific issues, see `PUBLISHING.md` for detailed troubleshooting.

