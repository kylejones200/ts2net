# tests/test_parity.py
import yaml, os
from tests.parity import ParityCase, run_parity_test as parity_one, format_parity_report as parity_report_text

R = os.environ.get("RSCRIPT", "Rscript")


def load_cases(path="tests/parity_config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for c in cfg:
        yield ParityCase(**c)


def test_parity_smoke():
    for case in load_cases():
        rep = parity_one(case, rscript=R, workdir=".parity")
        print(parity_report_text(rep))
        if case.kind != "DTW":
            assert rep.edges_jaccard >= 0.6  # tune as needed


# tests/parity_smoke.py
import numpy as np
from ts2net import RecurrenceNetwork, TransitionNetwork, graph_summary
from ts2net.stats import permutation_entropy, cao_e1_e2, false_nearest_neighbors, rqa_full


def test_rn_defaults():
    x = np.sin(np.linspace(0, 20, 2000))
    rn = RecurrenceNetwork(
        m=3,
        tau=2,
        rule="epsilon",
        target_density=0.08,
        theiler=1,
        metric="euclidean",
        sparse=True,
    )
    G, A = rn.fit_transform(x)
    s = graph_summary(G)
    assert s["n"] > 10


def test_tn_ordinal_ties():
    x = np.r_[np.ones(10), np.arange(50)]
    tn = TransitionNetwork(symbolizer="ordinal", order=3, delay=1, tie_rule="stable")
    G, A = tn.fit_transform(x)
    assert A.shape[0] > 0


def test_stats_suite():
    x = np.sin(np.linspace(0, 40, 4000))
    pe = permutation_entropy(x, order=4, delay=1)
    e1, e2 = cao_e1_e2(x, m_max=8, tau=2)
    fnn = false_nearest_neighbors(x, m_max=8, tau=2)
    assert 0.0 <= pe <= 1.0
    assert e1.size == 7 and e2.size == 6
    assert fnn.size == 7


def test_rqa_full():
    x = np.sin(np.linspace(0, 20, 2000))
    from ..core import embed

    X = embed(x, 3, 2)
    from ..core import RecurrenceNetwork

    rn = RecurrenceNetwork(m=3, tau=2, rule="epsilon", target_density=0.05)
    G, A = rn.fit_transform(x)
    rq = rqa_full(A, lmin=2, vmin=2)
    assert "RR" in rq and "DET" in rq
