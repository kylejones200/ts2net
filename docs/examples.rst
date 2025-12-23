Examples
========

The `examples/` directory contains comprehensive examples demonstrating ts2net functionality.

Notebooks
---------

Interactive Jupyter notebooks are available for hands-on learning:

.. toctree::
   :maxdepth: 1
   
   ../examples/quick_start.ipynb

Scripts
-------

Quick Start Example
-------------------

Basic introduction with synthetic data:

.. code-block:: bash

   python examples/quick_start.py

Shows:
- HVG (Horizontal Visibility Graph) construction
- NetworkX conversion
- Performance mode
- Comparing different methods

Real Economic Data Example
---------------------------

Comprehensive example using real economic data from FRED (Federal Reserve Economic Data):

.. code-block:: bash

   pip install pandas-datareader signalplot
   python examples/example_fred_data.py

This example demonstrates:
- Fetching correlated economic indicators (GDP, Unemployment Rate, CPI)
- Univariate analysis with visibility graphs (HVG/NVG)
- Proximity networks from sliding windows
- Multivariate network construction
- Recurrence and transition networks
- Network visualization

Generated visualizations are saved to ``examples/images/``.

Performance Benchmarks
----------------------

Compare Numba-accelerated vs. pure Python implementations:

.. code-block:: bash

   pip install numba
   python examples/benchmark_numba.py

Benchmarks:
- HVG performance
- NVG performance
- Recurrence network performance
- Transition network performance

Visualization Gallery
---------------------

The `viz_gallery.py <https://github.com/kylejones200/ts2net/tree/main/examples/viz_gallery.py>`_ example demonstrates all five flagship visualization functions on the same dataset:

1. **Time series with change points and window boundaries** - Shows the signal with detected change points and window edges
2. **Degree profile across time** - Plots degree versus time index as a proxy for local complexity
3. **Degree distribution as CCDF** - Complementary CDF on log scale for stable cross-zone comparison
4. **Method comparison panel** - Three aligned dot plots comparing edge count, average degree, and density
5. **Window level feature map** - Heatmap of window stats (mean degree, variance, etc.) for anomaly detection

All visualization functions are available in the `ts2net.viz` module:

.. code-block:: python

   from ts2net.viz import (
       plot_series_with_events,
       plot_degree_profile,
       plot_degree_ccdf,
       plot_method_comparison,
       plot_window_feature_map,
       plot_hvg_small,
       plot_recurrence_matrix,
   )

For more details, see the `examples/README.md <https://github.com/kylejones200/ts2net/tree/main/examples>`_ file.



