Transfer entropy
================

**Primary reference:** Schreiber (2000) *Phys. Rev. Lett.* **85**, 461.

Transfer entropy TE(X→Y) quantifies information flow from X to Y beyond Y's own
history. It is asymmetric: directed coupling should yield TE(X→Y) > TE(Y→X).

Assumptions
-----------

- Discrete binning or kNN estimation; bin count affects magnitude.
- Series should be long enough for stable density estimates.
- Coupled logistic maps in ``synthetic_causal`` validate asymmetry
  (``transfer_entropy_coupled_maps``).

API
---

- :func:`ts2net.causal.transfer_entropy.transfer_entropy`
- :func:`ts2net.causal.transfer_entropy.transfer_entropy_network`
- :func:`ts2net.causal.transfer_entropy.conditional_transfer_entropy`
