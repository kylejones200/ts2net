# PyPI Publishing Setup Guide

## Current Status

The PyPI publishing workflow is configured but requires manual setup on PyPI.

## Workflow Configuration

The workflow (`.github/workflows/publish-pypi.yml`) is set up to:
- Trigger on GitHub releases (when published)
- Trigger on version tags (e.g., `v0.5.0`)
- Allow manual trigger via `workflow_dispatch`

## Required Setup Steps

### 1. Configure PyPI Trusted Publisher

1. Go to https://pypi.org/manage/projects/
2. Select the `ts2net` project
3. Go to "Publishing" → "Add a new pending publisher"
4. Select "GitHub Actions" as the publisher type
5. Enter:
   - **Owner**: `kylejones200`
   - **Repository**: `ts2net`
   - **Workflow filename**: `publish-pypi.yml`
   - **Environment name**: `pypi` (must match the environment in the workflow)
6. Click "Add"

### 2. Create GitHub Environment (if not exists)

1. Go to https://github.com/kylejones200/ts2net/settings/environments
2. Create environment named `pypi`
3. Optionally add protection rules (required reviewers, etc.)

### 3. Publishing a Release

#### Option A: Via GitHub Release

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit and push changes
4. Create a git tag: `git tag v0.5.1 && git push origin v0.5.1`
5. Go to GitHub → Releases → "Draft a new release"
6. Select the tag you just created
7. Fill in release notes from CHANGELOG
8. Click "Publish release"
9. The workflow will automatically trigger and publish to PyPI

#### Option B: Via Tag Only

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit and push changes
4. Create a git tag: `git tag v0.5.1 && git push origin v0.5.1`
5. The workflow will automatically trigger on the tag push

#### Option C: Manual Trigger

1. Go to https://github.com/kylejones200/ts2net/actions/workflows/publish-pypi.yml
2. Click "Run workflow"
3. Select branch (usually `main`)
4. Click "Run workflow"

## Troubleshooting

### Workflow Not Triggering

- Check that the tag format is `v*` (e.g., `v0.5.0`, not `0.5.0`)
- Verify the workflow file is in `.github/workflows/publish-pypi.yml`
- Check GitHub Actions tab for any errors

### PyPI Authentication Errors

- Verify trusted publisher is configured correctly on PyPI
- Check that environment name matches (`pypi`)
- Ensure `id-token: write` permission is set in workflow

### Build Failures

- Check that Rust toolchain installs correctly
- Verify maturin can build the package locally: `maturin build --release`
- Check Python version compatibility (3.12+)

### Missing Wheels

- Verify all OS builds complete successfully
- Check artifact uploads in workflow logs
- Ensure `merge-multiple: true` in artifact download step

## Testing Locally

Before publishing, test the build locally:

```bash
# Install maturin
pip install maturin

# Build wheels
maturin build --release

# Check dist/ directory for built packages
ls -lh dist/
```

## Verification

After publishing, verify on PyPI:

1. Go to https://pypi.org/project/ts2net/
2. Check that the new version appears
3. Test installation: `pip install ts2net==0.5.1`
