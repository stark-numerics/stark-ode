from __future__ import annotations

import runpy


EXAMPLES = (
    "scalar_decay",
    "numpy_oscillator",
    "in_place_derivative",
    "choose_scheme",
    "checkpoints",
    "interface.native",
    "interface.numpy",
)


print("Getting started example runner")
print("==============================")

for example in EXAMPLES:
    module_name = f"examples.getting_started.{example}"
    print()
    print("=" * 80)
    print(f"Running {module_name}")
    print("=" * 80)
    runpy.run_module(module_name, run_name="__main__")

print()
print("All getting started examples completed.")

