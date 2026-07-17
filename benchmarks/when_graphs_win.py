#!/usr/bin/env python3
"""
Generate a narrative "when graphs win" report comparing feature sets.

Compares ts2net graph features vs statistical baselines on bundled UCR data.
Run:
    python benchmarks/when_graphs_win.py
    TS2NET_CI_SMOKE=1 python benchmarks/when_graphs_win.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ts2net.datasets.ucr import list_ucr_datasets, run_ucr_benchmark  # noqa: E402


def _interpret(payload: dict) -> list[str]:
    scores = payload.get("scores", {})
    ranked = sorted(
        scores.items(),
        key=lambda kv: kv[1]["mean_score"],
        reverse=True,
    )
    if not ranked:
        return ["No scores recorded."]

    lines: list[str] = []
    graph_keys = [
        k
        for k in scores
        if "graph" in k.lower() or "hvg" in k.lower() or "network" in k.lower()
    ]
    stat_keys = [k for k in scores if k not in graph_keys]

    if graph_keys and stat_keys:
        best_graph = max(graph_keys, key=lambda k: scores[k]["mean_score"])
        best_stat = max(stat_keys, key=lambda k: scores[k]["mean_score"])
        g_score = scores[best_graph]["mean_score"]
        s_score = scores[best_stat]["mean_score"]
        if g_score > s_score + 0.02:
            lines.append(
                f"Graph features (**{best_graph}**) beat the best statistical baseline "
                f"(**{best_stat}**) by {g_score - s_score:.3f} mean CV accuracy."
            )
            lines.append(
                "Graphs likely help: shape/topology carries class signal "
                "that scalar summaries miss."
            )
        elif s_score > g_score + 0.02:
            lines.append(
                f"Statistical baseline (**{best_stat}**) beats graph features "
                f"(**{best_graph}**) by {s_score - g_score:.3f}."
            )
            lines.append(
                "Scalar features may suffice — use graphs for interpretability "
                "or regime dynamics, not raw accuracy."
            )
        else:
            lines.append(
                f"Graph and statistical features are within 0.02 "
                f"({best_graph} vs {best_stat}). "
                "Choose based on explainability and downstream workflow."
            )
    else:
        best_name, best = ranked[0]
        lines.append(f"Best overall: **{best_name}** ({best['mean_score']:.3f}).")

    if "baseline_check" in payload:
        for msg in payload["baseline_check"].get("messages", []):
            lines.append(f"Baseline check: {msg}")

    return lines


def _dataset_section(payload: dict) -> str:
    lines = [
        f"### {payload['dataset']}",
        "",
        f"Source: {payload['metadata']['source']} · "
        f"CV folds: {payload.get('cv', 'n/a')}",
        "",
        "| Feature set | Mean score | Std | Features |",
        "| ----------- | ---------- | --- | -------- |",
    ]
    scores = payload.get("scores", {})
    ranked = sorted(
        scores.items(),
        key=lambda kv: kv[1]["mean_score"],
        reverse=True,
    )
    for name, s in ranked:
        lines.append(
            f"| {name} | {s['mean_score']:.3f} | {s['std_score']:.3f} | "
            f"{s['n_features']} |"
        )
    lines.append("")
    for line in _interpret(payload):
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def _rollup(payloads: list[dict]) -> str:
    graph_wins = 0
    stat_wins = 0
    ties = 0
    for payload in payloads:
        scores = payload.get("scores", {})
        graph_keys = [
            k
            for k in scores
            if "graph" in k.lower() or "hvg" in k.lower() or "network" in k.lower()
        ]
        stat_keys = [k for k in scores if k not in graph_keys]
        if not graph_keys or not stat_keys:
            continue
        g = max(scores[k]["mean_score"] for k in graph_keys)
        s = max(scores[k]["mean_score"] for k in stat_keys)
        if g > s + 0.02:
            graph_wins += 1
        elif s > g + 0.02:
            stat_wins += 1
        else:
            ties += 1

    lines = [
        "## Rollup",
        "",
        f"Datasets compared: **{len(payloads)}**",
        f"- Graph features win (by >0.02): **{graph_wins}**",
        f"- Statistical baselines win: **{stat_wins}**",
        f"- Within 0.02 (tie): **{ties}**",
        "",
    ]
    if graph_wins > stat_wins:
        lines.append(
            "Across bundled UCR panels, graph topology features often add signal for "
            "shape-heavy classification tasks."
        )
    elif stat_wins > graph_wins:
        lines.append(
            "On these panels, scalar statistical features are competitive — use graphs "
            "when you need interpretability, causal structure, or regime tracking."
        )
    else:
        lines.append(
            "Results are mixed — pick methods based on whether you need accuracy alone "
            "or explainable network evidence."
        )
    lines.append("")
    return "\n".join(lines)


def _narrative(payloads: list[dict]) -> str:
    lines = [
        "# When Graphs Win (and When They Don't)",
        "",
        "Multi-dataset comparison of ts2net graph features vs statistical baselines "
        "on bundled UCR classification archives.",
        "",
    ]
    lines.append(_rollup(payloads))
    lines.append("## Per-dataset results")
    lines.append("")
    for payload in payloads:
        lines.append(_dataset_section(payload))

    lines.extend(
        [
            "## Where to go next",
            "",
            "- Use `build_decision_package()` for human-readable decision evidence.",
            "- See `examples/recipes/` for domain workflows.",
            "- Full harness: `python benchmarks/run_ucr_benchmark.py`.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    smoke = os.environ.get("TS2NET_CI_SMOKE", "") == "1"
    out_dir = _REPO / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "when_graphs_win.json"
    md_path = out_dir / "when_graphs_win.md"

    datasets = list_ucr_datasets()
    if smoke:
        datasets = datasets[:2]

    payloads: list[dict] = []
    for name in datasets:
        per_json = out_dir / f"when_graphs_win_{name}.json"
        payload = run_ucr_benchmark(
            name,
            split="train",
            cv=3 if smoke else 5,
            include_optional_baselines=not smoke,
            output_path=per_json,
            validate_baselines=True,
        )
        payloads.append(payload)

    combined = {
        "datasets": [p["dataset"] for p in payloads],
        "timestamp_utc": payloads[0]["timestamp_utc"] if payloads else None,
        "results": payloads,
    }
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    narrative = _narrative(payloads)
    md_path.write_text(narrative, encoding="utf-8")

    print(narrative)
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
