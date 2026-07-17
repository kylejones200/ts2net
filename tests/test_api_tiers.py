"""Tests for v1.0 API stability tiers."""

from __future__ import annotations

import importlib

import pytest

import ts2net
from ts2net import api_tiers


@pytest.mark.parametrize("name", sorted(api_tiers.STABLE))
def test_stable_symbols_exported(name: str) -> None:
    assert hasattr(ts2net, name), f"{name} is stable but not exported from ts2net"


def test_api_tier_classification() -> None:
    assert api_tiers.api_tier("HVG") == "stable"
    assert api_tiers.api_tier("fit_sindy") == "experimental"
    assert api_tiers.api_tier("_private") == "internal"


def test_stable_and_experimental_disjoint() -> None:
    overlap = api_tiers.STABLE & api_tiers.EXPERIMENTAL
    assert not overlap, f"Symbols in both tiers: {overlap}"


def test_reports_module_matches_tiers() -> None:
    reports = importlib.import_module("ts2net.reports")
    for name in (
        "build_graph_report",
        "build_decision_package",
        "build_dynamic_change_report",
    ):
        assert name in api_tiers.STABLE
        assert hasattr(reports, name)
