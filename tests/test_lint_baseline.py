"""Tests for the lint-debt baseline gate.

The gate exists to stop inherited Ruff debt from growing. A gate that has never
been observed to fail is not a gate, so these tests drive it end to end against
throwaway projects: they add a violation and require failure, remove one and
require success.

The final test guards the committed baseline itself, so it cannot silently go
stale against the real tree.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "lint_baseline.py"

_spec = importlib.util.spec_from_file_location("lint_baseline", SCRIPT)
lint_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint_baseline)


RUFF_CONFIG = """\
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
"""

CLEAN = "value = 1\n"
ONE_UNUSED_IMPORT = "import os\n\nvalue = 1\n"
TWO_UNUSED_IMPORTS = "import os\nimport sys\n\nvalue = 1\n"


@pytest.fixture
def project(tmp_path):
    """A throwaway project with its own Ruff config and baseline."""
    (tmp_path / "pyproject.toml").write_text(RUFF_CONFIG)
    (tmp_path / "a.py").write_text(ONE_UNUSED_IMPORT)
    return tmp_path


def run(project_dir, *argv):
    """Invoke the gate against `project_dir`, returning its exit code."""
    return lint_baseline.main(
        [
            "--baseline",
            str(project_dir / ".lint-baseline.json"),
            "--root",
            str(project_dir),
            *argv,
        ]
    )


class TestBaselineCreation:
    def test_update_records_the_current_debt(self, project):
        assert run(project, "update") == 0
        data = json.loads((project / ".lint-baseline.json").read_text())
        assert data["total"] == 1
        assert data["files"]["a.py"] == {"F401": 1}
        assert data["ruff_version"].startswith("ruff ")

    def test_check_passes_when_nothing_changed(self, project, capsys):
        run(project, "update")
        assert run(project, "check") == 0
        assert "unchanged at 1" in capsys.readouterr().out


class TestGateFailsOnNewDebt:
    def test_an_extra_violation_in_a_dirty_file_fails(self, project, capsys):
        run(project, "update")
        (project / "a.py").write_text(TWO_UNUSED_IMPORTS)

        assert run(project, "check") == 1
        out = capsys.readouterr().out
        assert "Lint debt increased" in out
        assert "a.py:F401  baseline 1 -> now 2" in out

    def test_a_violation_in_a_brand_new_file_fails(self, project, capsys):
        run(project, "update")
        (project / "b.py").write_text(ONE_UNUSED_IMPORT)

        assert run(project, "check") == 1
        out = capsys.readouterr().out
        assert "b.py:F401  baseline 0 -> now 1" in out

    def test_a_different_rule_in_an_already_dirty_file_fails(self, project, capsys):
        """Being dirty under one rule is not a licence to add another."""
        run(project, "update")
        (project / "a.py").write_text(ONE_UNUSED_IMPORT + "\ndef F(x):\n    return x\n")

        assert run(project, "check") == 1
        assert "N802" in capsys.readouterr().out


class TestGatePassesOnImprovement:
    def test_removing_a_violation_passes_and_reports(self, project, capsys):
        run(project, "update")
        (project / "a.py").write_text(CLEAN)

        assert run(project, "check") == 0
        out = capsys.readouterr().out
        assert "Lint debt is DOWN: 1 -> 0" in out
        assert "lint-baseline-update" in out

    def test_swapping_one_violation_for_another_still_fails(self, project, capsys):
        """Total is unchanged, but the new finding is still new."""
        run(project, "update")
        (project / "a.py").write_text(CLEAN)
        (project / "b.py").write_text(ONE_UNUSED_IMPORT)

        assert run(project, "check") == 1
        assert "b.py:F401" in capsys.readouterr().out


class TestUpdateRefusesToRatchetUp:
    def test_update_will_not_record_an_increase(self, project, capsys):
        run(project, "update")
        (project / "a.py").write_text(TWO_UNUSED_IMPORTS)

        assert run(project, "update") == 1
        assert "Refusing to raise the baseline" in capsys.readouterr().out
        # The baseline on disk is untouched.
        assert json.loads((project / ".lint-baseline.json").read_text())["total"] == 1

    def test_an_increase_is_possible_only_deliberately(self, project):
        run(project, "update")
        (project / "a.py").write_text(TWO_UNUSED_IMPORTS)

        assert run(project, "update", "--allow-increase") == 0
        assert json.loads((project / ".lint-baseline.json").read_text())["total"] == 2


class TestReport:
    def test_report_lists_the_outstanding_debt(self, project, capsys):
        assert run(project, "report") == 0
        out = capsys.readouterr().out
        assert "Outstanding Ruff debt: 1 findings" in out
        assert "F401" in out


class TestCommittedBaselineIsCurrent:
    """The real repository must satisfy its own committed baseline."""

    def test_the_repo_passes_its_own_gate(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "the committed .lint-baseline.json is out of date with the tree:\n"
            + result.stdout
            + result.stderr
        )

    def test_the_baseline_file_is_committed_and_parseable(self):
        data = json.loads((REPO_ROOT / ".lint-baseline.json").read_text())
        assert data["total"] == sum(
            sum(rules.values()) for rules in data["files"].values()
        )
