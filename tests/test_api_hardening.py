"""
API hardening tests: validation, exceptions, and builder consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
import networkx as nx

from ts2net import (
    HVG,
    NVG,
    RecurrenceNetwork,
    TransitionNetwork,
    NetworkBuilder,
    NotBuiltError,
    ValidationError,
)
from ts2net._validation import (
    validate_series,
    validate_output_mode,
    validate_positive_int,
)


BUILDERS = [HVG, NVG, RecurrenceNetwork, TransitionNetwork]


class TestValidationHelpers:
    def test_validate_output_mode_rejects_invalid(self):
        with pytest.raises(ValidationError, match="output must be one of"):
            validate_output_mode("full", "HVG")

    def test_validate_positive_int_rejects_zero(self):
        with pytest.raises(ValidationError, match="must be >= 1"):
            validate_positive_int("order", 0, builder_name="TransitionNetwork")

    def test_validate_series_rejects_2d(self):
        with pytest.raises(ValidationError, match="1-D array"):
            validate_series(np.ones((3, 4)), "HVG")

    def test_validate_series_strips_nan(self):
        x = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
        clean = validate_series(x, "test", warn_degenerate=False)
        np.testing.assert_array_equal(clean, np.array([1.0, 2.0, 3.0]))


class TestBuilderProtocol:
    @pytest.mark.parametrize("cls", BUILDERS)
    def test_implements_network_builder(self, cls):
        builder = cls()
        assert isinstance(builder, NetworkBuilder)

    @pytest.mark.parametrize("cls", BUILDERS)
    def test_not_built_error_on_degree_sequence(self, cls):
        builder = cls()
        with pytest.raises(NotBuiltError, match="Network not built"):
            builder.degree_sequence()

    @pytest.mark.parametrize("cls", BUILDERS)
    def test_transform_without_fit(self, cls):
        builder = cls()
        with pytest.raises(NotBuiltError, match="Must call fit"):
            builder.transform()

    @pytest.mark.parametrize("cls", BUILDERS)
    def test_invalid_output_mode_at_init(self, cls):
        with pytest.raises(ValidationError, match="output must be one of"):
            cls(output="invalid")


class TestRecurrenceTransitionSklearnAPI:
    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (RecurrenceNetwork, {"rule": "knn", "k": 3}),
            (TransitionNetwork, {"order": 2}),
        ],
    )
    def test_fit_transform(self, cls, kwargs):
        x = np.random.randn(120)
        G = cls(**kwargs).fit_transform(x)
        assert isinstance(G, (nx.Graph, nx.DiGraph))
        assert G.number_of_nodes() > 0

    def test_constant_series_still_builds(self):
        x = np.ones(50)
        with pytest.warns(UserWarning, match="Constant series"):
            hvg = HVG().build(x)
        assert hvg.n_nodes == 50


class TestBuildNetworkFactory:
    def test_unknown_method(self):
        from ts2net import build_network

        with pytest.raises(ValueError, match="Unknown method"):
            build_network(np.ones(10), "unknown")

    @pytest.mark.parametrize("method", ["hvg", "nvg", "recurrence", "transition"])
    def test_all_methods(self, method):
        from ts2net import build_network

        x = np.random.randn(80)
        kwargs = {"rule": "knn", "k": 3} if method == "recurrence" else {}
        if method == "transition":
            kwargs = {"order": 2}
        g = build_network(x, method, **kwargs)
        assert g.n_nodes > 0
