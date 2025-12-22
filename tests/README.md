# Test Suite Documentation

## Test Organization

### Unit Tests (`tests/unit/`)

- **`test_hvg_correctness.py`**: Hard correctness tests comparing fast implementation vs naive O(n²) reference
- **`test_nvg_correctness.py`**: Hard correctness tests for NVG
- **`test_visibility_ties.py`**: Property-based tests for pathological ties (repeated values)
- **`test_visibility.py`**: Basic visibility graph tests

### Integration Tests (`tests/integration/`)

- **`test_pipeline_e2e.py`**: End-to-end tests for full pipeline (grouping, windowing, aggregation)

### Performance Tests (`tests/performance/`)

- **`test_hvg_performance.py`**: Performance benchmarks and regression tests

## Tie-Breaking Rules

### HVG (Horizontal Visibility Graph)

**Rule**: Two points i and j are horizontally visible if all points k between them satisfy `x[k] < min(x[i], x[j])`.

- If any intermediate point equals or exceeds `min(x[i], x[j])`, visibility is blocked
- Points with the same value are visible only if all intermediate points are strictly less
- Consecutive points are always visible (no intermediate points to block)

**Examples**:
- `[1, 2, 1]`: Points 0 and 2 (both value 1) are NOT visible (blocked by point 1 with value 2)
- `[2, 1, 2]`: Points 0 and 2 (both value 2) ARE visible (point 1 value 1 < 2)
- `[1, 1, 1]`: Points 0 and 2 are NOT visible (point 1 equals min(1, 1))

### NVG (Natural Visibility Graph)

**Rule**: Two points i and j are naturally visible if the line connecting `(i, x[i])` to `(j, x[j])` does not intersect any intermediate points.

- If an intermediate point k lies exactly on the line (`x[k] == line_height`), visibility is NOT blocked (implementation uses strict `>` comparison)
- Points below the line do not block visibility
- Points above the line block visibility

**Examples**:
- `[1, 2, 3]`: Points 0 and 2 ARE visible (point 1 is on line, but implementation allows it)
- `[1, 1.6, 2]`: Points 0 and 2 are NOT visible (point 1 is above line)

## Cross-Platform Determinism

Tests marked with `@pytest.mark.platform` should be run on both macOS and Linux in CI to verify:

1. Identical edge sets for the same input
2. No ordering-dependent bugs from parallel execution
3. Consistent results across NumPy/Rust implementations

The `TestCrossPlatformDeterminism` class contains fixtures that produce known edge set hashes that should match across platforms.

## Running Tests

```bash
# All tests
pytest

# Specific test categories
pytest tests/unit/test_visibility_ties.py  # Tie-breaking tests
pytest tests/integration/test_pipeline_e2e.py  # E2E pipeline tests
pytest tests/performance/test_hvg_performance.py  # Performance tests

# Cross-platform determinism (should run on both macOS and Linux)
pytest tests/unit/test_visibility_ties.py::TestCrossPlatformDeterminism -m platform
```

## Known Fixture Dataset

The `known_fixture_dataset` fixture in `test_pipeline_e2e.py` creates a predictable dataset:
- 3 series (A, B, C)
- Each with 100 points
- Series A: sine wave (periodic)
- Series B: linear trend + noise
- Series C: constant with spikes

Expected statistics are documented in `known_fixture_expected_stats` fixture for regression testing.
