from __future__ import annotations

from examples.manifest import examples_for_tier
from examples.runner import run_specs


print("Feature example runner")
print("======================")

run_specs(examples_for_tier("feature"))

print()
print("All feature examples completed.")
