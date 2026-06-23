from __future__ import annotations

from examples.manifest import examples_for_tier
from examples.runner import run_specs


print("Engines example runner")
print("======================")

run_specs(examples_for_tier("engines"))

print()
print("All engines examples completed.")
