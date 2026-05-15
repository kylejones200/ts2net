# Migration to uv

This project has been migrated from pip/requirements.txt to [uv](https://github.com/astral-sh/uv) for dependency management. This document provides a quick reference for the new commands.

## Installation

### Old (pip)
```bash
pip install -e ".[dev]"
```

### New (uv)
```bash
uv sync --group dev
```

## Running Commands

### Old (direct)
```bash
pytest
black ts2net/ tests/
flake8 ts2net/ tests/
python -m build
```

### New (uv run)
```bash
uv run pytest
uv run black ts2net/ tests/
uv run ruff check .
uv run ruff format .
uv run flake8 ts2net/ tests/
uv run python -m build
```

## Development Workflow

### Setup
```bash
# Clone repository
git clone https://github.com/kylejones200/ts2net.git
cd ts2net

# Install dependencies (creates .venv automatically)
uv sync --group dev

# Build Rust extension
uv run maturin develop --release
```

### Testing
```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=ts2net --cov-report=html

# Run specific test markers
uv run pytest -m slow
uv run pytest -m benchmark
```

### Linting and Formatting
```bash
# Check with ruff (recommended)
uv run ruff check .

# Format with ruff
uv run ruff format .

# Or use black/isort (still supported)
uv run black ts2net/ tests/
uv run isort ts2net/ tests/
uv run flake8 ts2net/ tests/
```

### Building
```bash
# Build package
uv run python -m build

# Check distribution
uv run twine check dist/*
```

## Key Changes

1. **Single source of truth**: `pyproject.toml` now contains all dependency information
2. **Lock file**: `uv.lock` provides reproducible builds with exact version pinning
3. **Virtual environment**: uv automatically manages `.venv` in the project directory
4. **Dependency groups**: Dev dependencies are in `[dependency-groups.dev]` (PEP 735)
5. **Legacy files**: `requirements-dev.txt` and `docs/requirements.txt` are kept for reference but should not be used

## Benefits

- **Faster**: uv is 10-100x faster than pip
- **Reproducible**: `uv.lock` ensures consistent builds across environments
- **Modern**: Uses PEP 621 (pyproject.toml) and PEP 735 (dependency groups)
- **Integrated**: Single tool for dependency management, virtual environments, and running commands

## Troubleshooting

### If uv is not installed
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### Sync dependencies after updating pyproject.toml
```bash
uv sync --group dev
```

### Update lock file
```bash
uv lock
```

### Clear cache and reinstall
```bash
rm -rf .venv
uv sync --group dev
```

## CI/CD

GitHub Actions workflows have been updated to use uv:
- Installs uv via `astral-sh/setup-uv@v5`
- Uses `uv sync --frozen` to respect the lock file
- Runs commands via `uv run`

## Further Reading

- [uv documentation](https://docs.astral.sh/uv/)
- [PEP 621 - Project metadata](https://peps.python.org/pep-0621/)
- [PEP 735 - Dependency groups](https://peps.python.org/pep-0735/)

