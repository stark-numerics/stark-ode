from __future__ import annotations

import numpy as np
import pytest

from benchmarks.fitzhugh_nagumo_1d.stark import FitzHughNagumoParameters, run_inverter_example
from stark import InverterBiCGStab, InverterFGMRES, InverterGMRES


@pytest.mark.slow
def test_fitzhugh_nagumo_inverters_track_each_other() -> None:
    parameters = FitzHughNagumoParameters(
        grid_size=32,
        t_stop=3.0,
        checkpoint_count=10,
        initial_step=1.0e-2,
        tolerance_atol=1.0e-6,
        tolerance_rtol=1.0e-5,
        resolution_atol=1.0e-7,
        resolution_rtol=1.0e-7,
        inversion_atol=1.0e-7,
        inversion_rtol=1.0e-7,
        resolution_max_iterations=20,
        inversion_max_iterations=20,
        inversion_restart=10,
    )

    trajectories = [
        run_inverter_example("GMRES", InverterGMRES, parameters),
        run_inverter_example("FGMRES", InverterFGMRES, parameters),
        run_inverter_example("BiCGStab", InverterBiCGStab, parameters),
    ]

    final_u = [trajectory.u_snapshots[-1] for trajectory in trajectories]
    final_v = [trajectory.v_snapshots[-1] for trajectory in trajectories]

    for trajectory in trajectories:
        assert trajectory.steps > 0
        assert np.isfinite(trajectory.runtime)
        assert np.all(np.isfinite(trajectory.u_snapshots[-1]))
        assert np.all(np.isfinite(trajectory.v_snapshots[-1]))

    for left in range(len(trajectories)):
        for right in range(left + 1, len(trajectories)):
            assert np.max(np.abs(final_u[left] - final_u[right])) < 5.0e-2
            assert np.max(np.abs(final_v[left] - final_v[right])) < 5.0e-2
