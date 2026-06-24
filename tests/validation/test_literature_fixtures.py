"""Literature validation fixtures (Horizon 9)."""

from __future__ import annotations

import os

import pytest

from benchmarks.literature_checks import load_literature_fixtures, run_fixture


def _fixture_ids() -> list[str]:
    smoke = os.environ.get("TS2NET_CI_SMOKE", "") == "1"
    return [spec["id"] for spec in load_literature_fixtures(smoke=smoke)]


@pytest.mark.validation
@pytest.mark.parametrize(
    "spec",
    load_literature_fixtures(smoke=os.environ.get("TS2NET_CI_SMOKE", "") == "1"),
    ids=_fixture_ids(),
)
def test_literature_fixture(spec):
    result = run_fixture(spec)
    assert result.passed, result.message
