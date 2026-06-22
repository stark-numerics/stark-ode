"""Shared helpers for runnable example groups."""

from __future__ import annotations

import runpy
from collections.abc import Iterable

from examples.manifest import ExampleSpec


def run_specs(specs: Iterable[ExampleSpec]) -> None:
    """Run example modules in manifest order."""

    for spec in specs:
        print()
        print("=" * 80)
        print(f"Running {spec.module}")
        print("=" * 80)
        runpy.run_module(spec.module, run_name="__main__")
