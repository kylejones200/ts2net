Visibility graphs
=================

**Primary reference:** Lacasa et al. (2008) *Proc. Natl. Acad. Sci.* **105**, 4972–4975.

Horizontal (HVG) and natural (NVG) visibility graphs connect time samples when
no intermediate sample blocks line-of-sight visibility in the time-amplitude plane.

Assumptions
-----------

- Samples are ordered and equally spaced (or treated as such).
- Mean HVG degree ≈ 4 for i.i.d. random series (literature fixture:
  ``hvg_mean_degree_gaussian``).
- Strictly monotone series form a consecutive chain (``hvg_monotone_edges``).

Directed irreversibility
------------------------

**Reference:** Lacasa et al. (2012) *Sci. Rep.* **2**, 378.

``visibility_irreversibility()`` compares forward vs reversed visibility statistics.
Asymmetric ramps should yield positive irreversibility scores.

API
---

- :class:`ts2net.api.HVG`
- :class:`ts2net.api.NVG`
- :func:`ts2net.causal.visibility.visibility_irreversibility`
