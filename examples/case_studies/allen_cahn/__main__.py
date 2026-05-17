from __future__ import annotations

# Allen-Cahn example runner
#
# In a source checkout, run the full lesson sequence from the `stark-ode`
# directory with:
#
#     python -m examples.case_studies.allen_cahn

import runpy


LESSONS = (
    "lesson_01_problem",
    "lesson_02_compare_explicit",
    "lesson_03_monitor_explicit",
    "lesson_04_implicit_newton",
    "lesson_05_imex_spectral",
    "lesson_06_compare_methods",
    "lesson_07_large_imex_run",
)


print("Allen-Cahn lesson runner")
print("========================")

for lesson in LESSONS:
    module_name = f"examples.case_studies.allen_cahn.{lesson}"
    print()
    print("=" * 80)
    print(f"Running {module_name}")
    print("=" * 80)
    runpy.run_module(module_name, run_name="__main__")

print()
print("All Allen-Cahn lessons completed.")
