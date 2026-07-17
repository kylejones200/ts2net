#!/usr/bin/env python3
"""Generate recipe notebooks from the domain recipe scripts."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "examples" / "recipes"

BOOTSTRAP = """from pathlib import Path
import sys

REPO_ROOT = Path.cwd()
if REPO_ROOT.name == "recipes":
    REPO_ROOT = REPO_ROOT.parent.parent
elif REPO_ROOT.name == "examples":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
"""

NOTEBOOKS: dict[str, dict[str, str]] = {
    "industrial_sensors.ipynb": {
        "title": "Industrial Sensors — Drift & Causal Drivers",
        "intro": (
            "Synthetic 3-sensor panel with injected drift on sensor 2.\n\n"
            "**Workflow:** data → causal graph → decision package\n\n"
            "Script: `uv run python examples/recipes/industrial_sensors.py`"
        ),
        "code": """import numpy as np

from ts2net.causal import CausalWorkflowSpec, run_causal_analysis
from ts2net.reports import build_decision_package

rng = np.random.default_rng(42)
n = 400
s0 = np.cumsum(rng.normal(0, 0.5, n))
s1 = 0.7 * np.roll(s0, 2) + rng.normal(0, 0.2, n)
s2 = rng.normal(0, 0.3, n)
s2[250:] += np.linspace(0, 3, n - 250)

X = np.vstack([s0, s1, s2])
names = ["motor_vibration", "bearing_temp", "line_pressure"]

causal = run_causal_analysis(
    X,
    spec=CausalWorkflowSpec(method="granger", max_lag=5, alpha=0.05, series_names=names),
)

pkg = build_decision_package(
    x=s2,
    method="hvg",
    window=50,
    causal=causal,
    title="Industrial sensor decision package",
)
print(pkg.to_markdown())""",
    },
    "energy_production.ipynb": {
        "title": "Energy Production — Well Analogs & Abnormal Decline",
        "intro": (
            "Synthetic well decline curves; one well has accelerated decline.\n\n"
            "**Workflow:** similarity graph → graph report → decision package"
        ),
        "code": """import numpy as np

from ts2net.graphs import similarity_network
from ts2net.reports import build_graph_report, build_decision_package

rng = np.random.default_rng(7)
n_wells, n_months = 8, 60
t = np.arange(n_months)
curves = []
names = []
for i in range(n_wells):
    q0 = 100 + 10 * i
    decline = 0.02 + 0.002 * i
    if i == 3:
        decline = 0.08
    q = q0 * np.exp(-decline * t) + rng.normal(0, 2, n_months)
    curves.append(q)
    names.append(f"well_{i}")

X = np.vstack(curves)
G, _ = similarity_network(X, method="dtw", threshold=0.35)
for i, name in enumerate(names):
    if G.has_node(i):
        G.nodes[i]["name"] = name

report = build_graph_report(
    G,
    method="similarity_dtw",
    parameters={"threshold": 0.35},
    title="Well analog network",
)
print(report.summary())
print()

deg = dict(G.degree())
isolated = [names[n] for n, d in deg.items() if d == 0]
if isolated:
    print(f"Wells with no analogs (review decline): {isolated}")

pkg = build_decision_package(
    curves[3],
    G=G,
    method="hvg",
    window=20,
    title="Abnormal decline well — decision package",
)
print()
print(pkg.summary())""",
    },
    "finance_regime.ipynb": {
        "title": "Finance — Regime Change & Rolling Correlation",
        "intro": (
            "Synthetic returns with a correlation breakdown mid-sample.\n\n"
            "**Workflow:** rolling graph decision package + correlation network"
        ),
        "code": """import numpy as np

from ts2net.graphs import rolling_correlation_network
from ts2net.reports import build_decision_package

rng = np.random.default_rng(99)
n = 500
r1 = rng.normal(0, 0.01, n)
r2 = 0.8 * r1 + rng.normal(0, 0.006, n)
r2[300:] = -0.5 * r1[300:] + rng.normal(0, 0.012, n - 300)
X = np.column_stack([r1, r2])

pkg = build_decision_package(
    r1,
    method="hvg",
    window=40,
    step=2,
    title="Finance regime — asset 1 rolling graph",
)
print(pkg.to_markdown())
print()
print("--- Rolling correlation network (last window) ---")
G = rolling_correlation_network(X[-80:], threshold=0.3)
print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")""",
    },
    "observability_services.ipynb": {
        "title": "Observability — Service Dependencies & Incident Precursors",
        "intro": (
            "Synthetic latency panel; service B degrades and perturbs C.\n\n"
            "**Workflow:** causal analysis → decision package"
        ),
        "code": """import numpy as np

from ts2net.causal import CausalWorkflowSpec, run_causal_analysis
from ts2net.reports import build_decision_package

rng = np.random.default_rng(12)
n = 300
a = np.abs(rng.normal(50, 5, n))
b = 0.6 * a + rng.normal(0, 3, n)
c = 0.4 * b + rng.normal(0, 2, n)
b[200:] += np.linspace(0, 40, n - 200)
c[220:] += np.linspace(0, 30, n - 220)

X = np.vstack([a, b, c])
names = ["api_gateway", "checkout_svc", "payment_db"]

causal = run_causal_analysis(
    X,
    spec=CausalWorkflowSpec(method="granger", max_lag=3, series_names=names),
)
pkg = build_decision_package(
    b,
    method="hvg",
    window=30,
    causal=causal,
    title="Observability incident precursor",
)
print(pkg.to_markdown())""",
    },
    "healthcare_trajectory.ipynb": {
        "title": "Healthcare — Patient Trajectory & Risk Shifts",
        "intro": (
            "Synthetic vitals with a late heart-rate risk shift.\n\n"
            "**Workflow:** dynamic visibility graphs → decision package"
        ),
        "code": """import numpy as np

from ts2net.reports import build_decision_package

rng = np.random.default_rng(21)
n = 240
hr = 72 + rng.normal(0, 3, n)
hr[160:] += np.linspace(0, 25, n - 160)
spo2 = 98 - 0.05 * (hr - 72) + rng.normal(0, 0.3, n)

pkg = build_decision_package(
    hr,
    method="hvg",
    window=35,
    step=3,
    title="Patient trajectory — heart rate graph shifts",
)
print(pkg.to_markdown())
print()
if pkg.dynamic_report:
    anom = pkg.dynamic_report.result.anomalous_windows()
    if len(anom):
        print(f"Review vitals at window indices: {anom.tolist()}")""",
    },
}


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _notebook(title: str, intro: str, code: str) -> dict:
    return {
        "cells": [
            _cell("markdown", f"# {title}\n\n{intro}"),
            _cell("code", BOOTSTRAP),
            _cell("code", code),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for filename, spec in NOTEBOOKS.items():
        path = RECIPES / filename
        nb = _notebook(spec["title"], spec["intro"], spec["code"])
        path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
