Recurrence networks and RQA
===========================

**Primary reference:** Marwan et al. (2007) *Phys. Rep.* **438**, 237–379.

Recurrence networks connect embedded phase-space points when their distance falls
below threshold ε. Recurrence quantification analysis (RQA) summarizes the
recurrence matrix (RR, DET, LAM, etc.).

Assumptions
-----------

- Embedding dimension ``m`` and delay ``tau`` affect topology; defaults are
  chosen for univariate smoke tests.
- Larger ε yields denser graphs (fixture: ``recurrence_epsilon_monotonicity``).
- Periodic signals yield high determinism DET (fixture: ``rqa_sine_determinism``).

API
---

- :class:`ts2net.core.recurrence.RecurrenceNetwork`
- :func:`ts2net.graphs.recurrence.recurrence_quantification`
- :func:`ts2net.stats.threshold_sensitivity.threshold_sensitivity_sweep`
