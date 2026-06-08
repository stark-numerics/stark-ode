from __future__ import annotations

import numpy as np
import pytest

from competition.fitzhugh_nagumo_1d.stark import FitzHughNagumoParameters, run_inverter_example
from stark.inverters.relaxation import InverterRelaxationRichardson


@pytest.mark.slow
def test_fitzhugh_nagumo_richardson_inverter_runs() -> None:
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

    trajectory = run_inverter_example("Richardson", InverterRelaxationRichardson, parameters)

    assert trajectory.steps > 0
    assert np.isfinite(trajectory.runtime)
    assert np.all(np.isfinite(trajectory.u_snapshots[-1]))
    assert np.all(np.isfinite(trajectory.v_snapshots[-1]))
