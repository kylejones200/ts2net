#!/usr/bin/env python3
"""Generate REPO_MAP.md for ts2net."""

import os
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_tree(root, max_depth=3, prefix="", depth=0):
    """Generate tree structure."""
    if depth >= max_depth:
        return []
    
    lines = []
    items = sorted(Path(root).iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    ignore = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 
              '.mypy_cache', 'dist', 'build', '.venv', 'venv', 'node_modules',
              '.parity', '.cursor'}
    
    items = [i for i in items if i.name not in ignore and not i.name.endswith('.pyc')]
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        lines.append(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and depth < max_depth - 1:
            lines.extend(get_tree(item, max_depth, prefix + next_prefix, depth + 1))
    
    return lines

def scan_modules():
    """Scan key modules and their purposes."""
    modules = {
        # Core implementations
        "core/visibility/hvg.py": "Horizontal Visibility Graph - O(n) stack algorithm with Numba",
        "core/visibility/nvg.py": "Natural Visibility Graph - Numba-accelerated",
        "core/recurrence.py": "Recurrence Networks - ε-ball and k-NN with fast distances",
        "core/transition.py": "Transition Networks - ordinal patterns and symbolization",
        "core/parallel.py": "Batch processing with multiprocessing",
        
        # Package implementations
        "ts2net/core/visibility.py": "Visibility graphs for package API",
        "ts2net/core/recurrence.py": "Recurrence networks for package API",
        "ts2net/core/transition.py": "Transition networks for package API",
        "ts2net/core/spatial.py": "Spatial weights (radius/k-NN)",
        
        # Network analysis
        "ts2net/networks/builders.py": "Graph construction utilities",
        "ts2net/networks/communities.py": "Community detection",
        "ts2net/networks/motifs.py": "Motif counting and analysis",
        "ts2net/networks/roles.py": "Node role identification",
        
        # Distance metrics
        "ts2net/distances/core.py": "Distance matrix computations",
        "ts2net/distances/dtw.py": "Dynamic Time Warping",
        
        # CLI
        "ts2net/cli/__init__.py": "Click-based command-line interface",
        "ts2net/cli/commands/network.py": "Network conversion commands",
        "ts2net/cli/commands/spatial.py": "Spatial analysis commands",
        
        # Rust bindings
        "ts2net_rs/src/lib.rs": "Rust core for performance-critical code",
        "ts2net_rs/src/graphs/visibility.rs": "Rust visibility graph implementations",
    }
    return modules

def main():
    """Generate REPO_MAP.md."""
    root = Path(__file__).parent.parent
    
    content = []
    
    # Header
    content.append("# ts2net Repository Map")
    content.append("")
    
    # Project summary
    content.append("## Project Summary")
    content.append("")
    content.append("Python library for converting time series to networks using multiple methods "
                   "(visibility graphs, recurrence networks, transition networks). "
                   "Heavily optimized with Numba JIT compilation (10-200x speedups) and optional Rust backend. "
                   "Includes CLI tools, batch processing, and comprehensive network analysis.")
    content.append("")
    
    # Primary goals
    content.append("## Primary Goals")
    content.append("")
    content.append("Transform univariate time series into graph representations for analysis. "
                   "Provide fast, production-ready implementations with automatic acceleration. "
                   "Maintain API compatibility with R ts2net package. "
                   "Support large-scale batch processing and distributed workflows.")
    content.append("")
    
    # Architecture
    content.append("## Architecture")
    content.append("")
    content.append("Three-layer design: (1) `core/` contains pure algorithmic implementations with Numba, "
                   "(2) `ts2net/` provides public API and integrations, "
                   "(3) `ts2net_rs/` offers optional Rust backend for critical paths. "
                   "All algorithms use dictionary dispatch patterns (no if/elif chains). "
                   "Automatic fallback from Numba→Python, Rust→Python.")
    content.append("")
    
    # Folder map
    content.append("## Folder Map")
    content.append("")
    content.append("```")
    content.append("ts2net/")
    tree_lines = get_tree(root, max_depth=3)
    for line in tree_lines[:80]:  # Limit output
        content.append(line)
    if len(tree_lines) > 80:
        content.append("... (truncated)")
    content.append("```")
    content.append("")
    
    # Key modules
    content.append("## Key Modules")
    content.append("")
    modules = scan_modules()
    for path, purpose in modules.items():
        content.append(f"- `{path}` - {purpose}")
    content.append("")
    
    # Entry points
    content.append("## Entry Points")
    content.append("")
    content.append("**CLI:** `ts2net` command (ts2net.cli:main)")
    content.append("```bash")
    content.append("ts2net to-parquet edges.csv --name graph --output out/")
    content.append("ts2net from-parquet graph_meta.json --graphml out.graphml")
    content.append("ts2net spatial-weights radius coords.txt --radius 1.0")
    content.append("ts2net spatial-weights knn coords.txt --k 5")
    content.append("```")
    content.append("")
    content.append("**Python API:**")
    content.append("```python")
    content.append("from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork")
    content.append("G, A = HVG().fit_transform(timeseries)")
    content.append("```")
    content.append("")
    
    # Public APIs
    content.append("## Public APIs")
    content.append("")
    content.append("**Graph Builders:**")
    content.append("- `HVG(weighted=False, sparse=False)` - Horizontal visibility")
    content.append("- `NVG(weighted=False, sparse=False)` - Natural visibility")
    content.append("- `RecurrenceNetwork(m, tau, rule, k, epsilon, metric)` - Phase space networks")
    content.append("- `TransitionNetwork(symbolizer, order, bins)` - State transition graphs")
    content.append("")
    content.append("**Batch Processing:**")
    content.append("- `batch_transform(X, builder, n_jobs=-1, **kwargs)` - Parallel processing")
    content.append("")
    content.append("**Analysis:**")
    content.append("- `graph_summary(G)` - Comprehensive graph metrics")
    content.append("- `motif_counts(G, motif_type)` - Motif enumeration")
    content.append("")
    
    # Data flow
    content.append("## Data Flow")
    content.append("")
    content.append("**Input:** 1D NumPy array (time series)")
    content.append("**Processing:** Graph construction (HVG/NVG/RN/TN) with Numba acceleration")
    content.append("**Output:** (NetworkX Graph, adjacency matrix)")
    content.append("**Analysis:** Metrics, motifs, communities, roles")
    content.append("")
    
    # Config
    content.append("## Config")
    content.append("")
    content.append("No global config. All parameters passed to constructors:")
    content.append("- `HVG(weighted, sparse)` - Graph construction options")
    content.append("- `RecurrenceNetwork(metric='euclidean'|'manhattan'|'chebyshev')` - Distance metrics")
    content.append("- `TransitionNetwork(symbolizer='ordinal'|'equal_width'|'equal_freq'|'kmeans')` - Symbolization")
    content.append("- `backend='rust'|'python'` - Backend selection (auto-fallback)")
    content.append("")
    
    # Runtime
    content.append("## Runtime")
    content.append("")
    content.append("**Python:** ≥3.9")
    content.append("**Core deps:** numpy, networkx, pandas, matplotlib")
    content.append("**Optional (speed):** numba (HIGHLY RECOMMENDED - 10-200x faster)")
    content.append("**Optional (features):** scipy, sklearn, tslearn, pyreadr")
    content.append("**Rust:** maturin for building Rust extensions (optional)")
    content.append("")
    
    # Tests
    content.append("## Tests")
    content.append("")
    content.append("**Run:** `pytest tests/`")
    content.append("**Structure:**")
    content.append("- `tests/unit/` - Unit tests for core functions")
    content.append("- `tests/integration/` - Integration tests")
    content.append("- `tests/test_numba_acceleration.py` - Numba-specific tests")
    content.append("**Fixtures:** `tests/conftest.py` with seeded random data")
    content.append("")
    
    # Examples
    content.append("## Examples")
    content.append("")
    content.append("**Basic usage:**")
    content.append("```python")
    content.append("import numpy as np")
    content.append("from ts2net import HVG, graph_summary")
    content.append("")
    content.append("x = np.sin(np.linspace(0, 12*np.pi, 800)) + 0.15*np.random.randn(800)")
    content.append("G, A = HVG().fit_transform(x)")
    content.append("print(graph_summary(G))")
    content.append("```")
    content.append("")
    content.append("**Benchmarks:** `python examples/benchmark_numba.py`")
    content.append("**Full examples:** `examples/example.py`, `examples/readme_plots.py`")
    content.append("")
    
    # Performance
    content.append("## Performance")
    content.append("")
    content.append("**With Numba:**")
    content.append("- HVG: 100x faster (O(n³)→O(n) + JIT)")
    content.append("- NVG: 180x faster (JIT compilation)")
    content.append("- Recurrence: 10x faster (distance computations)")
    content.append("- Ordinal patterns: 3x faster")
    content.append("")
    content.append("**Optimizations:**")
    content.append("- Dictionary dispatch (no if/elif chains)")
    content.append("- Numba JIT with caching")
    content.append("- Stack-based algorithms (HVG)")
    content.append("- Pre-allocated arrays")
    content.append("- Optional Rust backend")
    content.append("")
    
    # Known issues
    content.append("## Known Issues")
    content.append("")
    content.append("**Tech Debt:**")
    content.append("- Duplicate HVG/NVG implementations in `core/` and `ts2net/` (consolidation needed)")
    content.append("- Mixed import patterns (relative vs absolute)")
    content.append("- Some utils modules scattered across directories")
    content.append("")
    content.append("**Risks:**")
    content.append("- First Numba call includes compilation time (~1-2s)")
    content.append("- Large graphs (n>10000) may require sparse matrices")
    content.append("- Rust extensions optional but add build complexity")
    content.append("")
    
    # Recent changes
    content.append("## Recent Changes")
    content.append("")
    content.append("**Numba Acceleration (v0.4.0):**")
    content.append("- Added JIT compilation to all hot loops")
    content.append("- 10-200x speedups across algorithms")
    content.append("- Automatic fallback to pure Python")
    content.append("")
    content.append("**Refactoring:**")
    content.append("- Eliminated if/elif/else chains → dictionary dispatch")
    content.append("- Converted print statements → logging")
    content.append("- Simplified benchmark output")
    content.append("")
    
    # Write file
    output = root / "REPO_MAP.md"
    output.write_text("\n".join(content))
    logger.info(f"Generated {output}")
    logger.info(f"Lines: {len(content)}")

if __name__ == "__main__":
    main()

