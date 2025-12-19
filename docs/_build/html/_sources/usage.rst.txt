Usage Guide
==========

Install
-------

.. code-block:: bash

   pip install ts2net

Quick start
-----------

.. code-block:: python

   import numpy as np
   from ts2net import HVG, graph_summary

   x = np.sin(np.linspace(0, 12*np.pi, 800)) + 0.15 * np.random.randn(800)
   
   hvg = HVG()
   hvg.build(x)
   
   print(f"Nodes: {hvg.n_nodes}, Edges: {hvg.n_edges}")
   G = hvg.as_networkx()
   print(graph_summary(G))

CLI
---

.. code-block:: bash

   ts2net to-parquet --name my_graph --output out_dir edges.csv
   ts2net from-parquet --graphml out.graphml out_dir/graph_meta.json
   ts2net spatial-weights radius coords.txt --radius 1.0 --output weights.txt
   ts2net spatial-weights knn coords.txt --k 5 --output weights_knn.txt
