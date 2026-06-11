from __future__ import annotations

# Lesson 3: inspect one monitored explicit run
#
# The comparison report tells us whether methods agree and how long they took.
# Monitoring asks a different question: how did one solve behave internally?
#
# The point is to look past the final state. A solver can be accurate and still
# behave in a way that raises questions: many small steps, repeated rejections,
# or error ratios that sit far below the acceptance threshold.
#
# A useful adaptive-controller diagnostic is the error ratio over time:
#
# - below 1: accepted
# - around 1: right on the acceptance boundary
# - above 1: too large, so the trial step should be rejected
#
# As a rough rule of thumb, ratios around 0.1 to 0.5 suggest conservative
# stepping, while ratios around 0.5 to 1.0 suggest the controller is using more
# of the allowed local error budget. Repeated retries would suggest the
# controller is hunting too aggressively.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_03_monitor_explicit

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from stark import Configuration, Method
from stark.monitor import Monitor
from stark.methods.schemes import SchemeCashKarp

from examples.case_studies.allen_cahn.lesson_01_problem import (
    Configuration_TOLERANCE,
    Geometry,
    make_ivp,
)


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    geometry = Geometry()
    monitor = Monitor()

    # Monitoring is opt-in. We build the same explicit problem as lesson 1 and
    # pass the scheme monitor through the method recipe.

    ivp = make_ivp(
        geometry,
        method=Method(
            scheme=SchemeCashKarp,
            scheme_options={"monitor": monitor.scheme},
        ),
        configuration=Configuration(scheme_tolerance=Configuration_TOLERANCE),
    )

    ivp.final_result()

    # The monitor records one piece of evidence per accepted adaptive step.
    # Here we extract the time, local error ratio, and total rejected proposals.

    times = [step.t_end for step in monitor.scheme.adaptive_steps]
    error_ratios = [step.error_ratio for step in monitor.scheme.adaptive_steps]
    rejections = sum(step.rejection_count for step in monitor.scheme.adaptive_steps)
    max_error_ratio = max(error_ratios) if error_ratios else 0.0

    if max_error_ratio < 0.5:
        max_ratio_reading = (
            "even the largest accepted step used less than half the local "
            "error budget; this is conservative stepping."
        )
    elif max_error_ratio < 1.0:
        max_ratio_reading = (
            "the largest accepted step used a substantial fraction of the "
            "local error budget without crossing the acceptance boundary."
        )
    else:
        max_ratio_reading = (
            "at least one trial crossed the acceptance boundary; check the "
            "rejection count to see how often the controller retried."
        )

    print(f"accepted steps: {len(monitor.scheme.adaptive_steps)}")
    print(f"rejections:     {rejections}")
    print(f"max error ratio: {max_error_ratio:.4g}")
    print(f"reading:         {max_ratio_reading}")

    # Error ratios below one are accepted. Values far below one suggest the
    # adaptive controller is being conservative; spikes near one show where the
    # method is using most of the allowed local error.

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(times, error_ratios, marker=".", linewidth=1)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("error ratio")
    ax.set_title("Cash-Karp error-ratio trace")
    plot_path = HERE / "allen_cahn_cash_karp_error_ratio.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")
    print()
    print("What to notice:")
    print("- Error ratios below one are accepted steps; one is the acceptance boundary.")
    print("- If the maximum ratio is well below one, every accepted step was cautious.")
    print("- Conservative stepping does not mean the result is wrong, but it can be inefficient.")
    print("- That inefficiency becomes more important as the system size grows.")
