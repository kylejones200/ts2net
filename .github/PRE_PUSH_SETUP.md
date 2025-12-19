# Pre-Push Testing Setup

This document explains how to prevent pushing broken code to GitHub.

## Quick Start

**Before every push to `main` or `master`, run:**
```bash
make test-ci
# or
./scripts/test-local.sh
```

## What's Set Up

### 1. Git Pre-Push Hook (Automatic)
A pre-push hook is installed at `.git/hooks/pre-push` that:
- Automatically runs tests before pushing to `main` or `master`
- Blocks the push if tests fail
- Allows you to skip tests for other branches (but warns you)

**To disable temporarily:**
```bash
git push --no-verify
```
(Use with caution!)

### 2. Makefile Targets

- `make test` - Quick test run (fast, same as `pytest -q`)
- `make test-ci` - Full CI-like test run (slower, more thorough)
- `make check` - Alias for `test-ci` (good habit: `make check` before push)

### 3. Local Test Script

`scripts/test-local.sh` - Mimics the exact CI environment:
- Installs same dependencies as CI
- Runs same test command as CI
- Catches issues before they hit GitHub

## Recommended Workflow

1. **Before committing:**
   ```bash
   make test  # Quick check
   ```

2. **Before pushing to main:**
   ```bash
   make test-ci  # Full check
   git push
   ```

3. **If pre-push hook blocks you:**
   - Fix the failing tests locally
   - Run `make test-ci` again
   - Try pushing again

## Why This Matters

- **Faster feedback**: Catch errors in seconds, not minutes
- **Cleaner history**: No broken commits in git history
- **Better CI**: CI should verify, not debug
- **Team confidence**: Others can trust your pushes

## Troubleshooting

**Hook not running?**
```bash
chmod +x .git/hooks/pre-push
```

**Want to skip hook for one push?**
```bash
git push --no-verify  # Not recommended for main/master!
```

**Tests pass locally but fail in CI?**
- Check Python version: CI uses 3.12
- Check dependencies: CI installs specific versions
- Run `./scripts/test-local.sh` to match CI exactly
