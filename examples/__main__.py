from __future__ import annotations

from examples.manifest import EXAMPLES

print("STARK examples")
print("==============")
print("python -m examples.getting_started")
print("python -m examples.problem")
print("python -m examples.methods")
print("python -m examples.diagnostics")
print("python -m examples.engines")
print("python -m examples.inverters")
print("python -m examples.core")
print("python -m competition")
print()
print("Maintained runnable examples")
print("----------------------------")
for example in EXAMPLES:
    if example.default:
        suffix = ""
    elif example.cost == "cheap":
        suffix = " [manual]"
    else:
        suffix = f" [{example.cost}]"
    print(f"{example.module}{suffix}: {example.teaches}")
