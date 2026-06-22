"""Run the cheap backend examples."""

from __future__ import annotations

from examples.manifest import examples_for_tier
from examples.runner import run_specs


print("Backend example runner")
print("======================")

run_specs(examples_for_tier("backend"))

print()
print("All backend examples completed.")
