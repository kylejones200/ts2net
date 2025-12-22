Examples
========

The `examples/` directory contains comprehensive examples demonstrating ts2net functionality.

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

R API Parity Example
--------------------

Demonstrates R ts2net API parity for multivariate analysis:

.. code-block:: bash

   python examples/example_r_parity.py

Shows:
- Multiple time series → network
- Distance functions (correlation, DTW, NMI, VOI, etc.)
- Network builders (k-NN, ε-NN, weighted)
- Window-based proximity networks

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

For more details, see the `examples/README.md <https://github.com/kylejones200/ts2net/tree/main/examples>`_ file.


