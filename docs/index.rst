ts2net documentation
=====================

A Python implementation of **time series to network** methods for analyzing time series data through network science.

Overview
--------

ts2net converts time series into networks using various methods:

- **Visibility Graphs**: Horizontal Visibility Graph (HVG) and Natural Visibility Graph (NVG)
- **Recurrence Networks**: Phase space embedding with recurrence analysis
- **Transition Networks**: Symbolic dynamics and state transitions
- **Multivariate Networks**: Networks from multiple time series using distance metrics
- **Ordinal Partition Networks**: State space partitioning methods

Key Features
------------

- Fast implementations with Rust acceleration
- Multiple network construction methods
- Comprehensive visualization tools
- Multivariate time series support
- Windowed analysis for long time series
- Integration with NetworkX for network analysis

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from ts2net import HVG, graph_summary

   # Create a time series
   x = np.sin(np.linspace(0, 12*np.pi, 800)) + 0.15 * np.random.randn(800)
   
   # Build a Horizontal Visibility Graph
   hvg = HVG()
   hvg.build(x)
   
   # Get network statistics
   print(f"Nodes: {hvg.n_nodes}, Edges: {hvg.n_edges}")
   G = hvg.as_networkx()
   print(graph_summary(G))

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Guide

   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
