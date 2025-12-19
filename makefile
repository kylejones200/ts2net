# Makefile
.PHONY: dev docs test test-ci check

dev:
	pip install -U pip maturin
	maturin develop --release -m ts2net-rs/Cargo.toml
	pip install -e .[speed,dtw,approx,rdata]

test:
	@echo "Running quick tests..."
	pytest -q

test-ci:
	@echo "Running CI-like test suite (same as GitHub Actions)..."
	@echo "This may take longer but catches more issues."
	PYTHONHASHSEED=0 pytest -q

check: test-ci
	@echo "✅ All checks passed!"

docs:
	pip install mkdocs-material
	mkdocs serve -a 0.0.0.0:8000
