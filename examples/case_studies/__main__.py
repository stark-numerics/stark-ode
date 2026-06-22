from __future__ import annotations

from examples.manifest import EXAMPLES

print("Case study examples")
print("===================")
for example in EXAMPLES:
    if example.tier == "case-study":
        print(f"python -m {example.module}")
