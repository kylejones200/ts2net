# Contributing to ts2net

Thank you for your interest in contributing to ts2net! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kylejones200/ts2net.git
   cd ts2net
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install Rust (for building extensions):**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. **Build the Rust extension:**
   ```bash
   maturin develop --release
   ```

### Running Tests

```bash
# Run fast tests (default)
pytest

# Run all tests including slow/benchmark
pytest -m "not (slow or benchmark)"  # Fast tests only
pytest -m slow                        # Slow tests
pytest -m benchmark                   # Benchmark tests
pytest -m hard_validation             # Hard validation tests

# With coverage
pytest --cov=ts2net --cov-report=html
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black ts2net/ tests/`
- Use `isort` for import sorting: `isort ts2net/ tests/`
- Use `flake8` for linting: `flake8 ts2net/ tests/`
- Type hints are encouraged but not required

### Testing

- **Write tests for new features** - Aim for high coverage
- **Test invariants, not implementation details** - Tests should pass even if internals change
- **Use fixed random seeds** - All tests should be deterministic
- **Mark slow tests** - Use `@pytest.mark.slow` for tests that take >1 second
- **Mark benchmark tests** - Use `@pytest.mark.benchmark` for performance tests

### Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: add support for weighted visibility graphs

- Add weighted parameter to HVG and NVG
- Implement distance-based edge weights
- Add tests for weighted graph construction
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure tests pass

3. **Update documentation** if needed (README, docstrings, etc.)

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results
   - Any breaking changes

### Code Review

- All PRs require review before merging
- Address review comments promptly
- Keep PRs focused and reasonably sized
- Update CHANGELOG.md for user-facing changes

## Project Structure

```
ts2net/
├── ts2net/              # Main package
│   ├── api.py          # Graph builder classes (HVG, NVG, etc.)
│   ├── core/           # Core graph construction algorithms
│   ├── multivariate/   # Multivariate time series support
│   ├── viz/            # Visualization functions
│   └── ...
├── ts2net_rs/          # Rust extension (performance-critical code)
├── tests/              # Test suite
├── examples/          # Example scripts and notebooks
└── docs/              # Sphinx documentation
```

## Areas for Contribution

### High Priority

- **Performance improvements** - Especially for large time series
- **Additional network methods** - New graph construction algorithms
- **Documentation** - Examples, tutorials, API docs
- **Bug fixes** - See GitHub issues

### Medium Priority

- **More examples** - Real-world use cases
- **Visualization improvements** - Better plotting functions
- **Error handling** - More helpful error messages
- **Type hints** - Improve type coverage

### Low Priority

- **Code cleanup** - Refactoring, consolidation
- **Test coverage** - Increase coverage in under-tested areas
- **CI/CD improvements** - Build and test automation

## Questions?

- Open an issue for bug reports or feature requests
- Check existing issues and discussions
- Review the documentation at https://ts2net.readthedocs.io

Thank you for contributing to ts2net!


