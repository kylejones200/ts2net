"""
Runner for parity tests between R and Python implementations.
"""
from __future__ import annotations
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import numpy as np

from .models import ParityCase, ParityReport
from .utils import nx_from_python_case, compare_graphs, read_r_graphml


def run_r_case(case: ParityCase, rscript: str = "Rscript", workdir: str = ".parity") -> Path:
    """Run a single test case using R and return the output directory."""
    case_dir = Path(workdir) / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Write case configuration
    cjson = case_dir / "case.json"
    cjson.write_text(
        json.dumps(
            {
                "kind": case.kind,
                "series": case.series,
                "panel": case.panel,
                "params": case.params or {},
            },
            indent=2,
        )
    )
    
    # Run R script
    out_dir = case_dir / "r"
    out_dir.mkdir(exist_ok=True)
    
    script_path = Path(__file__).parent.parent.parent / "scripts" / "r" / "ts2net_case.R"
    cmd = [
        rscript,
        str(script_path),
        str(cjson),
        str(out_dir),
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"R script failed: {res.stderr}")
        
    return out_dir


def run_parity_test(
    case: ParityCase, 
    rscript: str = "Rscript", 
    workdir: str = ".parity"
) -> ParityReport:
    """Run a single parity test case and return the results."""
    out_r = run_r_case(case, rscript, workdir)
    
    if case.kind == "DTW":
        # Handle DTW comparison separately
        Dr = np.loadtxt(out_r / "dtw.csv", delimiter=",", skiprows=1)
        import pandas as pd
        from ts2net_rs import cdist_dtw as _cdist_dtw_rs

        Dp = _cdist_dtw_rs(
            (
                pd.read_csv(case.panel).to_numpy(float)
                if "name" not in pd.read_csv(case.panel).columns
                else pd.read_csv(case.panel).drop(columns=["name"]).to_numpy(float)
            ),
            band=(case.params or {}).get("band"),
        )
        rel = float(np.nanmean(np.abs(Dp - Dr) / np.maximum(1e-12, np.abs(Dr))))
        return ParityReport(
            case.name, 
            case.kind, 
            edges_jaccard=None,
            deg_l1=None,
            tri_rel_err=None,
            C_rel_err=None,
            L_rel_err=None,
            notes=f"DTW rel err={rel:.3e}"
        )
    
    # For graph-based comparisons
    Gr = read_r_graphml(out_r / "graph.graphml")
    Gp = nx_from_python_case(case)
    jacc, deg_l1, tri_rel, C_rel, L_rel = compare_graphs(Gr, Gp)
    
    return ParityReport(
        name=case.name,
        kind=case.kind,
        edges_jaccard=jacc,
        deg_l1=deg_l1,
        tri_rel_err=tri_rel,
        C_rel_err=C_rel,
        L_rel_err=L_rel,
        notes=""
    )


def format_parity_report(rep: ParityReport) -> str:
    """Format a parity report as a human-readable string."""
    lines = []
    lines.append(f"Case: {rep.name} [{rep.kind}]")
    
    if rep.kind == "DTW":
        lines.append(rep.notes)
        return "\n".join(lines)
    
    lines.append(f"Edge Jaccard: {rep.edges_jaccard:.4f}")
    lines.append(f"Degree L1 mean: {rep.deg_l1:.4f}")
    lines.append(f"Triangles rel err: {rep.tri_rel_err:.4f}")
    lines.append(f"C rel err: {rep.C_rel_err:.4f}")
    lines.append(f"L rel err: {rep.L_rel_err:.4f}")
    
    return "\n".join(lines)
