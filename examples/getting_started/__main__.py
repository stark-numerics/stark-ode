from __future__ import annotations

from examples.manifest import examples_for_tier
from examples.runner import run_specs


print("Getting started example runner")
print("==============================")

run_specs(examples_for_tier("getting-started"))

print()
print("All getting started examples completed.")
