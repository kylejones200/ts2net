"""
Literature validation checks for Horizon 9 (shared by CLI and pytest).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "literature.json"


@dataclass
class FixtureResult:
    """Outcome of a single literature fixture check."""

    id: str
    method: str
    passed: bool
    message: str
    citation: str = ""
    observed: dict[str, Any] | None = None


def _asymmetric_ramp(n: int) -> np.ndarray:
    half = n // 2
    return np.concatenate(
        [
            np.linspace(0.0, 10.0, half, dtype=np.float64),
            np.zeros(n - half, dtype=np.float64),
        ]
    )


def _check_hvg_mean_degree(params: dict[str, Any]) -> FixtureResult:
    from ts2net import HVG

    n_trials = int(params.get("n_trials", 200))
    n_points = int(params.get("n_points", 5000))
    seed = int(params.get("seed", 42))
    lo = float(params.get("min_mean", 3.85))
    hi = float(params.get("max_mean", 4.15))

    rng = np.random.default_rng(seed)
    builder = HVG(output="stats")
    degrees = []
    for _ in range(n_trials):
        x = rng.standard_normal(n_points)
        builder.build(x)
        degrees.append(float(builder.stats()["avg_degree"]))
    mean_deg = float(np.mean(degrees))
    passed = lo <= mean_deg <= hi
    return FixtureResult(
        id="",
        method="hvg_mean_degree",
        passed=passed,
        message=f"mean avg_degree={mean_deg:.4f} (expected [{lo}, {hi}])",
        observed={"mean_avg_degree": mean_deg, "n_trials": n_trials},
    )


def _check_hvg_edge_count(params: dict[str, Any]) -> FixtureResult:
    from ts2net import HVG

    n = int(params.get("n_points", 20))
    expected = int(params.get("expected_edges", n - 1))
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    builder = HVG(output="stats").build(x)
    n_edges = int(builder.stats()["n_edges"])
    passed = n_edges == expected
    return FixtureResult(
        id="",
        method="hvg_edge_count",
        passed=passed,
        message=f"n_edges={n_edges} (expected {expected})",
        observed={"n_edges": n_edges},
    )


def _check_visibility_irreversibility(params: dict[str, Any]) -> FixtureResult:
    from ts2net.causal.visibility import visibility_irreversibility

    n = int(params.get("n_points", 120))
    min_score = float(params.get("min_score", 0.0))
    x = _asymmetric_ramp(n)
    score = float(visibility_irreversibility(x))
    passed = score > min_score
    return FixtureResult(
        id="",
        method="visibility_irreversibility",
        passed=passed,
        message=f"irreversibility_score={score:.4f} (expected > {min_score})",
        observed={"irreversibility_score": score},
    )


def _check_transfer_entropy_asymmetry(params: dict[str, Any]) -> FixtureResult:
    from ts2net.causal.transfer_entropy import transfer_entropy
    from ts2net.datasets import load_dataset

    n = int(params.get("n", 3000))
    seed = int(params.get("seed", 42))
    coupling = float(params.get("coupling", 0.15))
    min_diff = float(params.get("min_diff", 0.1))

    data = load_dataset("synthetic_causal", n=n, seed=seed, coupling=coupling)
    x, y = data["X"].T
    te_xy = float(transfer_entropy(x, y, lag=1))
    te_yx = float(transfer_entropy(y, x, lag=1))
    diff = te_xy - te_yx
    passed = diff >= min_diff
    return FixtureResult(
        id="",
        method="transfer_entropy_asymmetry",
        passed=passed,
        message=f"TE(x→y)-TE(y→x)={diff:.4f} (expected >= {min_diff})",
        observed={"te_xy": te_xy, "te_yx": te_yx, "diff": diff},
    )


def _check_recurrence_threshold_monotonicity(params: dict[str, Any]) -> FixtureResult:
    from ts2net.core.recurrence import RecurrenceNetwork

    n = int(params.get("n_points", 200))
    thresholds = [float(t) for t in params.get("thresholds", [0.05, 0.2, 0.5, 1.0, 2.0])]
    x = np.sin(np.linspace(0.0, 8.0 * np.pi, n, dtype=np.float64))
    edge_counts: list[int] = []
    for eps in thresholds:
        rn = RecurrenceNetwork(rule="epsilon", threshold=eps)
        G, _ = rn.fit_transform(x)
        edge_counts.append(int(G.number_of_edges()))
    monotonic = all(
        edge_counts[i] <= edge_counts[i + 1] for i in range(len(edge_counts) - 1)
    )
    return FixtureResult(
        id="",
        method="recurrence_threshold_monotonicity",
        passed=monotonic,
        message=f"n_edges vs epsilon={edge_counts} (expected non-decreasing)",
        observed={"thresholds": thresholds, "n_edges": edge_counts},
    )


def _check_spain_hvg_mean_degree(params: dict[str, Any]) -> FixtureResult:
    from ts2net.datasets import load_dataset

    lo = float(params.get("min_mean", 3.95))
    hi = float(params.get("max_mean", 4.05))
    try:
        data = load_dataset("spain_meters_summary")
    except FileNotFoundError as exc:
        if params.get("optional", True):
            return FixtureResult(
                id="",
                method="spain_hvg_mean_degree",
                passed=True,
                message=f"skipped: {exc}",
                observed={"skipped": True},
            )
        return FixtureResult(
            id="",
            method="spain_hvg_mean_degree",
            passed=False,
            message=str(exc),
        )

    mean_deg = float(np.mean(data["X"][:, 0]))
    passed = lo <= mean_deg <= hi
    return FixtureResult(
        id="",
        method="spain_hvg_mean_degree",
        passed=passed,
        message=f"mean hvg_avg_degree={mean_deg:.4f} (expected [{lo}, {hi}])",
        observed={"mean_hvg_avg_degree": mean_deg, "n_meters": data["metadata"]["n_meters"]},
    )


_CHECKERS = {
    "hvg_mean_degree": _check_hvg_mean_degree,
    "hvg_edge_count": _check_hvg_edge_count,
    "visibility_irreversibility": _check_visibility_irreversibility,
    "transfer_entropy_asymmetry": _check_transfer_entropy_asymmetry,
    "recurrence_threshold_monotonicity": _check_recurrence_threshold_monotonicity,
    "spain_hvg_mean_degree": _check_spain_hvg_mean_degree,
}


def load_literature_fixtures(
  path: Path | None = None,
  *,
  smoke: bool = False,
) -> list[dict[str, Any]]:
    """Load fixture definitions from JSON."""
    path = path or _FIXTURES_PATH
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    fixtures = list(data.get("fixtures", []))
    if smoke:
        # Fast CI subset: skip Monte Carlo mean-degree and slow TE tests
        skip_methods = {"hvg_mean_degree", "transfer_entropy_asymmetry"}
        fixtures = [f for f in fixtures if f.get("method") not in skip_methods]
    return fixtures


def run_fixture(spec: dict[str, Any]) -> FixtureResult:
    """Run one literature fixture and return the result."""
    method = spec["method"]
    if method not in _CHECKERS:
        return FixtureResult(
            id=spec.get("id", method),
            method=method,
            passed=False,
            message=f"unknown method {method!r}",
            citation=spec.get("citation", ""),
        )
    result = _CHECKERS[method](spec.get("params", {}))
    result.id = spec.get("id", method)
    result.citation = spec.get("citation", "")
    return result


def run_all_fixtures(*, smoke: bool = False) -> list[FixtureResult]:
    """Run every literature fixture."""
    return [run_fixture(spec) for spec in load_literature_fixtures(smoke=smoke)]


def write_validation_manifest(
    results: list[FixtureResult],
    path: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Write reproducibility manifest JSON and return the payload."""
    try:
        import ts2net

        version = getattr(ts2net, "__version__", "unknown")
    except ImportError:
        version = "unknown"

    payload = {
        "ts2net_version": version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "passed": all(r.passed for r in results),
        "n_checks": len(results),
        "n_failed": sum(1 for r in results if not r.passed),
        "checks": [
            {
                "id": r.id,
                "method": r.method,
                "passed": r.passed,
                "message": r.message,
                "citation": r.citation,
                "observed": r.observed,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
