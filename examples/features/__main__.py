from __future__ import annotations

import runpy


EXAMPLES = (
    "manual_marcher_setup",
    "custom_scheme_fixed_explicit",
    "structured_state_minimal",
    "compare_two_schemes",
    "monitor_scheme_steps",
    "compare_with_monitor_summary",
    "inverter_request_and_defect",
    "inverter_relaxation_richardson",
    "inverter_relaxation_jacobi",
    "inverter_relaxation_specialist",
)


print("Feature example runner")
print("======================")

for example in EXAMPLES:
    module_name = f"examples.features.{example}"
    print()
    print("=" * 80)
    print(f"Running {module_name}")
    print("=" * 80)
    runpy.run_module(module_name, run_name="__main__")

print()
print("All feature examples completed.")
