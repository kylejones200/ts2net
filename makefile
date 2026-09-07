# Makefile
.PHONY: dev docs test test-ci rust-test rust-check check lint format

dev:
	uv sync --group dev
	uv run maturin develop --release -m ts2net_rs/Cargo.toml
	uv pip install numba tslearn pynndescent pyreadr || true

test:
	@echo "Running quick tests..."
	uv run pytest -q

test-ci:
	@echo "Running CI-like test suite (same as GitHub Actions)..."
	@echo "This may take longer but catches more issues."
	PYTHONHASHSEED=0 uv run pytest -q

rust-test:
	@echo "Running Rust unit tests..."
	# --no-default-features drops the PyO3 bindings. A test binary cannot link
	# against the symbols pyo3's extension-module feature leaves undefined, so a
	# plain `cargo test` fails at the link step; the algorithm modules under test
	# have no Python dependency anyway.
	cargo test --manifest-path ts2net_rs/Cargo.toml --no-default-features

rust-check:
	@echo "Checking both feature configurations..."
	cargo check --manifest-path ts2net_rs/Cargo.toml --no-default-features
	cargo check --manifest-path ts2net_rs/Cargo.toml

check: test-ci rust-test lint
	@echo "All checks passed!"

lint:
	@echo "Running linters..."
	uv run ruff check .
	uv run flake8 ts2net/ tests/ || true

format:
	@echo "Formatting code..."
	uv run ruff format .
	uv run black ts2net/ tests/ || true
	uv run isort ts2net/ tests/ || true

docs:
	uv pip install mkdocs-material
	uv run mkdocs serve -a 0.0.0.0:8000
