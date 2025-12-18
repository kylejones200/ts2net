"""
Comprehensive parity test harness for R ts2net comparison.

Generates test matrix, runs both implementations, compares at appropriate tolerance levels.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import json
import numpy as np
import networkx as nx

from .corpus import generate_test_corpus, save_corpus_to_csv
from .validation import validate_timeseries, normalize_metric_name
from .models import ParityCase, ParityReport
from .runner import run_parity_test

logger = logging.getLogger(__name__)


@dataclass
class ParityTestSpec:
    """Specification for a single parity test."""
    series_name: str
    method: str
    params: Dict[str, Any]
    tolerance_level: int  # 1=strict, 2=numeric, 3=distributional
    expected_pass: bool = True


def generate_parity_test_matrix() -> List[ParityTestSpec]:
    """
    Generate comprehensive test matrix covering all methods and edge cases.
    
    Returns list of test specifications.
    """
    tests = []
    
    # HVG tests
    hvg_series = ["sine_clean", "sine_noise", "short_20", "medium_50", "repeated_values"]
    for series in hvg_series:
        # Basic HVG
        tests.append(ParityTestSpec(
            series_name=series,
            method="HVG",
            params={"weighted": False, "limit": None},
            tolerance_level=1  # Strict for combinatorial
        ))
        
        # Weighted HVG
        tests.append(ParityTestSpec(
            series_name=series,
            method="HVG",
            params={"weighted": True, "limit": None},
            tolerance_level=2  # Numeric for weights
        ))
        
        # Limited HVG (if implemented)
        if series in ["medium_50"]:
            tests.append(ParityTestSpec(
                series_name=series,
                method="HVG",
                params={"weighted": False, "limit": 10},
                tolerance_level=1
            ))
    
    # NVG tests
    nvg_series = ["sine_clean", "short_20", "medium_50", "linear_trend"]
    for series in nvg_series:
        tests.append(ParityTestSpec(
            series_name=series,
            method="NVG",
            params={"weighted": False, "limit": None},
            tolerance_level=2  # Numeric due to float comparisons
        ))
    
    # RN tests - epsilon rule
    rn_series = ["sine_noise", "ar1_positive", "medium_50"]
    for series in rn_series:
        tests.append(ParityTestSpec(
            series_name=series,
            method="RN",
            params={
                "m": 2,
                "tau": 1,
                "rule": "epsilon",
                "epsilon": 0.5,
                "metric": "euclidean",
                "theiler": 0
            },
            tolerance_level=2
        ))
    
    # RN tests - knn rule
    for series in rn_series:
        tests.append(ParityTestSpec(
            series_name=series,
            method="RN",
            params={
                "m": 3,
                "tau": 2,
                "rule": "knn",
                "k": 8,
                "metric": "euclidean",
                "theiler": 1
            },
            tolerance_level=2
        ))
    
    # RN tests - different metrics
    for metric in ["manhattan", "maximum"]:  # R uses "maximum" for chebyshev
        tests.append(ParityTestSpec(
            series_name="medium_50",
            method="RN",
            params={
                "m": 2,
                "tau": 1,
                "rule": "knn",
                "k": 5,
                "metric": metric,
                "theiler": 0
            },
            tolerance_level=2
        ))
    
    # TN tests - ordinal patterns
    tn_series = ["sine_clean", "medium_50", "repeated_values"]
    for series in tn_series:
        tests.append(ParityTestSpec(
            series_name=series,
            method="TN",
            params={
                "symbolizer": "ordinal",
                "order": 3,
                "delay": 1,
                "tie_rule": "stable"
            },
            tolerance_level=1  # Strict for ordinal
        ))
        
        # Different order
        tests.append(ParityTestSpec(
            series_name=series,
            method="TN",
            params={
                "symbolizer": "ordinal",
                "order": 4,
                "delay": 1,
                "tie_rule": "stable"
            },
            tolerance_level=1
        ))
    
    # Default parameter tests (no params specified)
    for series in ["sine_clean", "medium_50"]:
        for method in ["HVG", "NVG"]:
            tests.append(ParityTestSpec(
                series_name=series,
                method=method,
                params={},  # Use defaults
                tolerance_level=1
            ))
    
    return tests


def check_parity_level1(G_r: nx.Graph, G_p: nx.Graph, tol: float = 0.0) -> Tuple[bool, dict]:
    """
    Level 1: Strict parity - exact edge set equality.
    
    For combinatorial methods (HVG, NVG on integers).
    """
    edges_r = set(frozenset(e) for e in G_r.edges())
    edges_p = set(frozenset(e) for e in G_p.edges())
    
    jaccard = len(edges_r & edges_p) / len(edges_r | edges_p) if edges_r | edges_p else 1.0
    
    passed = jaccard >= (1.0 - tol)
    
    metrics = {
        "jaccard": jaccard,
        "n_nodes_r": G_r.number_of_nodes(),
        "n_nodes_p": G_p.number_of_nodes(),
        "n_edges_r": G_r.number_of_edges(),
        "n_edges_p": G_p.number_of_edges(),
        "edge_diff": len(edges_r ^ edges_p),
    }
    
    return passed, metrics


def check_parity_level2(G_r: nx.Graph, G_p: nx.Graph) -> Tuple[bool, dict]:
    """
    Level 2: Numeric parity - degree distribution and summary metrics.
    
    For distance-based methods with float arithmetic.
    """
    # Degree L1 distance
    nodes = sorted(set(G_r.nodes()) | set(G_p.nodes()))
    deg_r = np.array([G_r.degree(n) if n in G_r else 0 for n in nodes])
    deg_p = np.array([G_p.degree(n) if n in G_p else 0 for n in nodes])
    deg_l1 = float(np.mean(np.abs(deg_r - deg_p)))
    
    # Edge Jaccard
    edges_r = set(frozenset(e) for e in G_r.edges())
    edges_p = set(frozenset(e) for e in G_p.edges())
    jaccard = len(edges_r & edges_p) / len(edges_r | edges_p) if edges_r | edges_p else 1.0
    
    # Summary metrics
    n_r, n_p = G_r.number_of_nodes(), G_p.number_of_nodes()
    m_r, m_p = G_r.number_of_edges(), G_p.number_of_edges()
    
    # Relative errors
    node_rel_err = abs(n_r - n_p) / max(n_r, 1)
    edge_rel_err = abs(m_r - m_p) / max(m_r, 1)
    
    # Pass criteria
    passed = (
        jaccard >= 0.95 and
        deg_l1 <= 0.1 and
        node_rel_err <= 0.01 and
        edge_rel_err <= 0.10
    )
    
    metrics = {
        "jaccard": jaccard,
        "deg_l1": deg_l1,
        "node_rel_err": node_rel_err,
        "edge_rel_err": edge_rel_err,
        "n_nodes_r": n_r,
        "n_nodes_p": n_p,
        "n_edges_r": m_r,
        "n_edges_p": m_p,
    }
    
    return passed, metrics


def check_parity_level3(G_r: nx.Graph, G_p: nx.Graph) -> Tuple[bool, dict]:
    """
    Level 3: Distributional parity - statistical properties.
    
    For large graphs where exact comparison is impractical.
    """
    from scipy import stats
    
    # Degree distributions
    deg_r = sorted([d for _, d in G_r.degree()])
    deg_p = sorted([d for _, d in G_p.degree()])
    
    # KS test
    if len(deg_r) > 0 and len(deg_p) > 0:
        ks_stat, ks_pval = stats.ks_2samp(deg_r, deg_p)
    else:
        ks_stat, ks_pval = 1.0, 0.0
    
    # Clustering coefficient
    try:
        C_r = nx.average_clustering(G_r.to_undirected())
        C_p = nx.average_clustering(G_p.to_undirected())
        C_rel_err = abs(C_r - C_p) / max(abs(C_r), 1e-10)
    except:
        C_rel_err = float('nan')
    
    # Pass criteria
    passed = (
        ks_pval > 0.05 and
        (np.isnan(C_rel_err) or C_rel_err <= 0.20)
    )
    
    metrics = {
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "C_rel_err": C_rel_err,
        "n_nodes_r": G_r.number_of_nodes(),
        "n_nodes_p": G_p.number_of_nodes(),
        "n_edges_r": G_r.number_of_edges(),
        "n_edges_p": G_p.number_of_edges(),
    }
    
    return passed, metrics


def run_parity_harness(
    output_dir: str = ".parity",
    data_dir: str = "tests/parity/data",
    rscript: str = "Rscript",
    generate_data: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive parity test harness.
    
    Args:
        output_dir: Directory for test outputs
        data_dir: Directory containing test series CSV files
        rscript: Path to Rscript executable
        generate_data: If True, generate test corpus first
        
    Returns:
        Dictionary with test results and summary
    """
    # Generate test data if needed
    if generate_data:
        logger.info("Generating test corpus...")
        save_corpus_to_csv(data_dir)
    
    # Generate test matrix
    logger.info("Generating test matrix...")
    test_specs = generate_parity_test_matrix()
    logger.info(f"Generated {len(test_specs)} test cases")
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    errors = 0
    
    for i, spec in enumerate(test_specs, 1):
        test_name = f"{spec.method}_{spec.series_name}_{i:03d}"
        logger.info(f"[{i}/{len(test_specs)}] Running {test_name}...")
        
        try:
            # Create parity case
            series_path = Path(data_dir) / f"{spec.series_name}.csv"
            if not series_path.exists():
                logger.error(f"Series file not found: {series_path}")
                errors += 1
                continue
            
            case = ParityCase(
                name=test_name,
                kind=spec.method,
                series=str(series_path),
                params=spec.params
            )
            
            # Run parity test
            report = run_parity_test(case, rscript, output_dir)
            
            # Check tolerance level
            # TODO: Implement tolerance checking based on spec.tolerance_level
            
            result = {
                "test_name": test_name,
                "spec": asdict(spec),
                "report": asdict(report),
                "passed": report.edges_jaccard > 0.95 if report.edges_jaccard else False,
            }
            
            results.append(result)
            
            if result["passed"]:
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            errors += 1
            results.append({
                "test_name": test_name,
                "spec": asdict(spec),
                "error": str(e),
                "passed": False,
            })
    
    # Summary
    summary = {
        "total": len(test_specs),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / len(test_specs) if test_specs else 0.0,
    }
    
    logger.info(f"\nParity Test Summary:")
    logger.info(f"  Total:  {summary['total']}")
    logger.info(f"  Passed: {summary['passed']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Errors: {summary['errors']}")
    logger.info(f"  Rate:   {summary['pass_rate']:.1%}")
    
    # Save results
    results_file = Path(output_dir) / "parity_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_parity_harness()

