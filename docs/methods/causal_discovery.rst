Causal discovery (PCMCI-style)
==============================

**Primary reference:** Runge et al. (2019) *Nature Communications* **10**, 2553.

``time_lagged_causality_network()`` evaluates transfer entropy or Granger
causality at multiple lags — a PCMCI-style building block for lagged discovery.

Assumptions
-----------

- Panel of aligned univariate series.
- Lagged driver X→Y should dominate reverse flow at the true lag
  (fixture: ``pcmci_lagged_driver``).
- Full PCMCI with conditioning sets is not yet bundled; see PC/FCI adapters.

API
---

- :func:`ts2net.causal.time_lagged.time_lagged_causality_network`
- :func:`ts2net.causal.pc.pc_algorithm`
- :func:`ts2net.causal.fci.fci_algorithm`
