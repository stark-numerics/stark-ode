from __future__ import annotations

from examples.manifest import examples_for_tier
from examples.runner import run_specs


print("Core example runner")
print("===================")

run_specs(examples_for_tier("core"))

print()
print("All core examples completed.")
