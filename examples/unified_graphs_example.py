"""
Example: Unified Graph Construction API

Demonstrates the unified TSGraph API for recurrence networks,
ordinal partition networks, and visibility graphs as described
in Varley & Sporns (2020) style construction.

All three graph types use the same TSGraph container and draw_tsgraph renderer.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from ts2net.viz import (
    build_recurrence_graph,
    build_ordinal_partition_graph,
    build_visibility_graph,
    draw_tsgraph,
    optimal_lag,
    optimal_dim,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def example_recurrence_epsilon_sweep():
    """Demonstrate ε-recurrence networks with different epsilon values."""
    logger.info("Example 1: Recurrence Networks (ε-sweep)")
    
    # Create a simple time series
    t = np.linspace(0, 4*np.pi, 100)
    x = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Build recurrence graphs with different epsilon values
    epsilons = [0.1, 0.2, 0.3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, eps in enumerate(epsilons):
        tsgraph = build_recurrence_graph(
            x,
            embed_dim=3,
            delay=1,
            eps=eps,
            eps_mode='fraction_max',
            return_pos=True,
        )
        
        draw_tsgraph(
            tsgraph,
            ax=axes[idx],
            show=False,
            node_size=8,
            edge_alpha=0.1,
            color_by='time',
        )
        axes[idx].set_title(f'ε = {eps} ({tsgraph.graph.number_of_edges()} edges)', fontsize=11)
    
    plt.suptitle('Recurrence Networks: Epsilon Sweep', fontsize=13)
    plt.tight_layout()
    output_path = Path(__file__).parent / 'images' / 'recurrence_epsilon_sweep.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created {len(epsilons)} recurrence graphs with different epsilon values")


def example_ordinal_partition():
    """Demonstrate ordinal partition network."""
    logger.info("Example 2: Ordinal Partition Network")
    
    # Create a time series with clear patterns
    t = np.linspace(0, 4*np.pi, 150)
    x = np.sin(t) + 0.05 * np.random.randn(150)
    
    # Use optimal parameters
    tau = optimal_lag(x)
    d = optimal_dim(x, delay=tau, dim_range=(3, 6))
    
    tsgraph = build_ordinal_partition_graph(
        x,
        embed_dim=d,
        delay=tau,
        directed=True,
        weighted=True,
        return_pos=True,
    )
    
    fig, ax = draw_tsgraph(
        tsgraph,
        show=False,
        node_size=15,
        edge_alpha=0.3,
        color_by='degree',
        cmap='plasma',
    )
    ax.set_title(f'Ordinal Partition Network (d={d}, τ={tau})\n{tsgraph.graph.number_of_nodes()} patterns, {tsgraph.graph.number_of_edges()} transitions', 
                 fontsize=11)
    
    output_path = Path(__file__).parent / 'images' / 'ordinal_partition.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created OPN with {tsgraph.graph.number_of_nodes()} patterns")


def example_visibility_comparison():
    """Compare HVG and NVG using unified API."""
    logger.info("Example 3: Visibility Graphs Comparison")
    
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
    
    plt.suptitle('Visibility Graphs: HVG vs NVG', fontsize=13)
    plt.tight_layout()
    output_path = Path(__file__).parent / 'images' / 'visibility_comparison.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"HVG: {tsgraph_hvg.graph.number_of_edges()} edges")
    logger.info(f"NVG: {tsgraph_nvg.graph.number_of_edges()} edges")


if __name__ == "__main__":
    logger.info("Unified Graph Construction API Examples")
    logger.info("Based on Varley & Sporns (2020) style construction")
    
    example_recurrence_epsilon_sweep()
    example_ordinal_partition()
    example_visibility_comparison()
    
    logger.info("All examples completed!")
    logger.info("Check examples/images/ for generated PNG files.")
