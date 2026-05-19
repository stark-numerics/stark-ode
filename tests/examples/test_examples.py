from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        (("examples",), "STARK examples"),
        (("examples.getting_started",), "All getting started examples completed."),
        (("examples.features",), "All feature examples completed."),
        (("examples.case_studies",), "Case study examples"),
        (("examples.comparison",), "Comparison examples"),
        (("examples.comparison.check_reports", "--help"), "Run comparison reports"),
    ),
)
def test_cheap_examples_run_without_traceback(
    command: tuple[str, ...],
    expected: str,
) -> None:
    module, *arguments = command
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Traceback" not in output
    assert expected in output
