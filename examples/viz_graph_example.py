"""
Example: Unified Graph Visualization API

Demonstrates the new TSGraph and draw_tsgraph API for creating
publication-quality visualizations of time-series-derived graphs.
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from ts2net.viz import build_visibility_graph, draw_tsgraph


def example_basic_hvg():
    """Basic HVG visualization."""
    print("Example 1: Basic HVG")
    
    # Create a simple time series
    t = np.linspace(0, 4*np.pi, 100)
    x = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Build graph
    tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
    
    # Draw with default settings (colored by time)
    fig, ax = draw_tsgraph(tsgraph, show=False, node_size=20)
    ax.set_title('HVG: Sine Wave (colored by time)', fontsize=12)
    plt.savefig('hvg_basic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Created graph with {tsgraph.graph.number_of_nodes()} nodes, "
          f"{tsgraph.graph.number_of_edges()} edges")


def example_directed_hvg():
    """Directed HVG for irreversibility analysis."""
    print("\nExample 2: Directed HVG")
    
    # Create asymmetric signal (ramp up, then drop)
    x = np.concatenate([
        np.linspace(0, 10, 50),  # Ramp up
        np.array([0] * 50)       # Sudden drop
    ])
    
    # Build directed graph
    tsgraph = build_visibility_graph(x, kind='hvg', directed=True, weighted=True)
    
    # Draw with degree coloring
    fig, ax = draw_tsgraph(
        tsgraph, 
        show=False, 
        node_size=15,
        color_by='degree',
        cmap='plasma'
    )
    ax.set_title('Directed HVG: Ramp + Drop (colored by degree)', fontsize=12)
    plt.savefig('hvg_directed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Created directed graph with {tsgraph.graph.number_of_edges()} edges")


def example_weight_modes():
    """Different weight modes for edge weights."""
    print("\nExample 3: Weight Modes")
    
    x = np.sin(np.linspace(0, 4*np.pi, 80)) + 0.1 * np.random.randn(80)
    
    weight_modes = ['absdiff', 'time_gap', 'slope']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, weight_mode in enumerate(weight_modes):
        tsgraph = build_visibility_graph(
            x, 
            kind='hvg', 
            weighted=weight_mode,
            directed=False
        )
        
        # Draw on subplot
        draw_tsgraph(
            tsgraph,
            ax=axes[idx],
            show=False,
            node_size=10,
            color_by='time'
        )
        axes[idx].set_title(f'Weight: {weight_mode}', fontsize=11)
    
    plt.suptitle('HVG with Different Weight Modes', fontsize=13)
    plt.tight_layout()
    plt.savefig('hvg_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Created graphs with {len(weight_modes)} different weight modes")


def example_nvg_vs_hvg():
    """Compare NVG vs HVG."""
    print("\nExample 4: NVG vs HVG Comparison")
    
    x = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # HVG
    tsgraph_hvg = build_visibility_graph(x, kind='hvg', directed=False)
    draw_tsgraph(tsgraph_hvg, ax=axes[0], show=False, node_size=15, color_by='time')
    axes[0].set_title(f'HVG ({tsgraph_hvg.graph.number_of_edges()} edges)', fontsize=11)
    
    # NVG
    tsgraph_nvg = build_visibility_graph(x, kind='nvg', directed=False)
    draw_tsgraph(tsgraph_nvg, ax=axes[1], show=False, node_size=15, color_by='time')
    axes[1].set_title(f'NVG ({tsgraph_nvg.graph.number_of_edges()} edges)', fontsize=11)
    
    plt.suptitle('HVG vs NVG Comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig('hvg_vs_nvg.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  HVG: {tsgraph_hvg.graph.number_of_edges()} edges")
    print(f"  NVG: {tsgraph_nvg.graph.number_of_edges()} edges")


def example_epsilon_sweep_style():
    """Epsilon sweep style visualization (multiple graphs)."""
    print("\nExample 5: Multiple Graphs (Epsilon Sweep Style)")
    
    x = np.sin(np.linspace(0, 4*np.pi, 60)) + 0.2 * np.random.randn(60)
    
    # Create multiple graphs with different limits (simulating epsilon changes)
    limits = [None, 10, 5]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, limit in enumerate(limits):
        tsgraph = build_visibility_graph(
            x,
            kind='hvg',
            directed=False,
            limit=limit
        )
        
        draw_tsgraph(
            tsgraph,
            ax=axes[idx],
            show=False,
            node_size=12,
            color_by='time'
        )
        
        limit_str = 'None' if limit is None else str(limit)
        axes[idx].set_title(f'Limit={limit_str}\n({tsgraph.graph.number_of_edges()} edges)', 
                           fontsize=10)
    
    plt.suptitle('HVG with Different Temporal Limits', fontsize=13)
    plt.tight_layout()
    plt.savefig('hvg_limits.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Created {len(limits)} graphs with different limits")


if __name__ == "__main__":
    print("=" * 60)
    print("Unified Graph Visualization API Examples")
    print("=" * 60)
    
    example_basic_hvg()
    example_directed_hvg()
    example_weight_modes()
    example_nvg_vs_hvg()
    example_epsilon_sweep_style()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)
