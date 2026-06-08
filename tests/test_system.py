from __future__ import annotations

import numpy as np

from stark import Configuration, Interval, StarkLayout, StarkLayoutField, StarkMethod, StarkSystem
from stark.interface.derivative import StarkDerivativeTimeInPlace
from stark.accelerators import AcceleratorNone
from stark.engines import StarkEngineNumpy


class ExplicitScheme:
    def __init__(
        self,
        derivative,
        allocator,
        *,
        configuration=None,
        specialist=None,
    ) -> None:
        self.derivative = derivative
        self.allocator = allocator
        self.configuration = configuration
        self.specialist = specialist

    def snapshot_state(self, state):
        out = self.allocator.allocate_state()
        self.allocator.copy_state(state, out)
        return out

    def __call__(self, interval, state) -> float:
        del interval, state
        return 0.0


def derivative(interval, state, out) -> None:
    del interval, state, out


def test_system_ivp_builds_engine_state_and_declared_scheme() -> None:
    layout = StarkLayout(
        (
            StarkLayoutField("u", translation="du", shape=(2,)),
            StarkLayoutField("v", translation="dv", shape=(2,)),
        )
    )
    system = StarkSystem(derivative=derivative, layout=layout)
    configuration = Configuration(check_progress=False)
    factory_layouts = []

    def engine_factory(layout):
        factory_layouts.append(layout)
        return StarkEngineNumpy(layout, accelerator=AcceleratorNone())

    ivp = system.ivp(
        initial={
            "u": np.array([1.0, 2.0]),
            "v": np.array([3.0, 4.0]),
        },
        interval=object(),
        method=StarkMethod(scheme=ExplicitScheme),
        engine=engine_factory,
        configuration=configuration,
    )

    assert factory_layouts == [layout]
    assert isinstance(ivp.scheme, ExplicitScheme)
    assert isinstance(ivp.scheme.derivative, StarkDerivativeTimeInPlace)
    assert ivp.scheme.derivative.function is derivative
    assert ivp.scheme.allocator is ivp.engine.allocator
    assert ivp.scheme.configuration is configuration
    assert ivp.scheme.specialist is ivp.engine.algebraist_specialist
    np.testing.assert_array_equal(ivp.initial.u, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(ivp.initial.v, np.array([3.0, 4.0]))


def test_system_ivp_trajectory_helpers_start_from_fresh_working_objects() -> None:
    layout = StarkLayout({"u": {"translation": "du", "shape": (1,)}})
    system = StarkSystem(derivative=derivative, layout=layout)

    ivp = system.ivp(
        initial={"u": np.array([1.0])},
        interval=Interval(0.0, 0.1, 0.0),
        method=StarkMethod(scheme=ExplicitScheme),
        engine=lambda layout: StarkEngineNumpy(layout, accelerator=AcceleratorNone()),
        configuration=Configuration(check_progress=False),
    )

    first_state = ivp.fresh_state()
    second_state = ivp.fresh_state()
    first_interval = ivp.fresh_interval()
    second_interval = ivp.fresh_interval()

    assert first_state is not ivp.initial
    assert first_state is not second_state
    assert first_interval is not ivp.interval
    assert first_interval is not second_interval
    np.testing.assert_array_equal(first_state.u, np.array([1.0]))

    first_state.u[0] = 2.0
    assert second_state.u[0] == 1.0
