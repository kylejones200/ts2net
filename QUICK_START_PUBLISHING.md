# Quick Start: Automatic PyPI Publishing

Your project is now configured for **automatic PyPI publishing** when you create a GitHub release!

## One-Time Setup (5 minutes)

### Step 1: Set Up PyPI Trusted Publishing

1. **Create a PyPI account** (if you don't have one):
   - Go to https://pypi.org/account/register/
   - Verify your email

2. **Set up Trusted Publishing** (no API tokens needed!):
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in exactly:
     ```
     PyPI Project Name: ts2net
     Owner: kylejones200
     Repository name: ts2net
     Workflow name: publish-pypi.yml
     Environment name: (leave blank)
     ```
   - Click "Add"

3. **Done!** No tokens to manage, no secrets to configure.

### Step 2: Set Up ReadTheDocs (Optional)

1. Go to https://readthedocs.org/accounts/signup/
2. Sign up and connect your GitHub account
3. Import your project: https://readthedocs.org/dashboard/import/
4. Select `kylejones200/ts2net`
5. Click "Next" and "Finish"

That's it! Docs will build automatically on every push.

## How to Publish a New Version

### Every time you want to release:

```bash
# 1. Update version in pyproject.toml
#    Change: version = "0.5.0"  (or whatever new version)

# 2. Update CHANGELOG.md with what's new

# 3. Commit the changes
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.5.0"
git push

# 4. Create and push a tag
git tag v0.5.0
git push origin v0.5.0

# 5. Create a GitHub Release
#    Go to: https://github.com/kylejones200/ts2net/releases/new
#    - Select tag: v0.5.0
#    - Title: v0.5.0
#    - Description: Copy from CHANGELOG.md
#    - Click "Publish release"
```

### What Happens Automatically:

1. ✅ GitHub Actions triggers
2. ✅ Builds wheels for Linux, macOS, Windows
3. ✅ Builds for Python 3.9, 3.10, 3.11, 3.12
4. ✅ Uploads everything to PyPI
5. ✅ ReadTheDocs builds new documentation
6. ✅ Package is live at https://pypi.org/project/ts2net/

**Total time: ~10-15 minutes** (most of it is waiting for builds)

## Verify Everything Works

Before your first release, test the setup:

```bash
# Run validation script
python scripts/validate_release.py

# Test local build
pip install maturin
maturin build --release

# Check that wheels were created
ls -lh dist/
```

## Your First Release Checklist

- [x] URLs updated with kylejones200 username ✅
- [ ] PyPI trusted publishing configured
- [ ] ReadTheDocs project imported (optional but recommended)
- [ ] Validation script passes
- [ ] Ready to create v0.4.0 release!

## After First Release

Once published, anyone can install with:
```bash
pip install ts2net
```

Updates are just as easy - repeat the release steps above with a new version number!

## Need Help?

- **Detailed guide**: See `PUBLISHING.md`
- **Step-by-step checklist**: See `SETUP_CHECKLIST.md`
- **PyPI Help**: https://pypi.org/help/
- **GitHub Actions logs**: https://github.com/kylejones200/ts2net/actions

## Pro Tips

1. **Use semantic versioning**: MAJOR.MINOR.PATCH (e.g., 0.5.0, 1.0.0, 1.1.0)
2. **Test locally first**: Run `maturin build --release` before tagging
3. **Keep CHANGELOG.md updated**: Users appreciate knowing what changed
4. **Create releases from GitHub UI**: Easier than command line
5. **Monitor first build**: Watch the Actions tab to ensure everything works

---

**You're all set!** The hardest part (configuration) is done. Publishing is now just:
update version → commit → tag → create release → done! 🚀

