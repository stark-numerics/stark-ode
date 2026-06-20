from __future__ import annotations

import numpy as np

from stark import Configuration, Interval, Frame, FrameField, Method, System
from stark.problem.derivative.derivative import DerivativeAdapterAcceptsInstant
from stark.engines.shared.accelerators import AcceleratorNone
from stark.engines import EngineNumpy


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
        del state
        return min(interval.step, interval.stop - interval.present)


class ImplicitScheme(ExplicitScheme):
    def __init__(
        self,
        derivative,
        allocator,
        resolvent,
        *,
        configuration=None,
        specialist=None,
    ) -> None:
        super().__init__(
            derivative,
            allocator,
            configuration=configuration,
            specialist=specialist,
        )
        self.resolvent = resolvent


def derivative(interval, state, out) -> None:
    del interval, state, out


def test_system_ivp_builds_engine_state_and_declared_scheme() -> None:
    frame = Frame(
        (
            FrameField("u", translation="du", shape=(2,)),
            FrameField("v", translation="dv", shape=(2,)),
        )
    )
    system = System(derivative=derivative, frame=frame)
    configuration = Configuration(check_progress=False)
    factory_layouts = []

    def engine_factory(frame):
        factory_layouts.append(frame)
        return EngineNumpy(frame, accelerator=AcceleratorNone())

    ivp = system.ivp(
        initial={
            "u": np.array([1.0, 2.0]),
            "v": np.array([3.0, 4.0]),
        },
        interval=object(),
        method=Method(scheme=ExplicitScheme),
        engine=engine_factory,
        configuration=configuration,
    )

    assert factory_layouts == [frame]
    assert isinstance(ivp.scheme, ExplicitScheme)
    assert isinstance(ivp.scheme.derivative, DerivativeAdapterAcceptsInstant)
    assert ivp.scheme.derivative.function is derivative
    assert ivp.scheme.allocator is ivp.engine.allocator
    assert ivp.scheme.configuration is configuration
    assert ivp.scheme.specialist is ivp.engine.algebraist_specialist
    np.testing.assert_array_equal(ivp.initial.u, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(ivp.initial.v, np.array([3.0, 4.0]))


def test_system_ivp_trajectory_helpers_start_from_fresh_working_objects() -> None:
    frame = Frame({"u": {"translation": "du", "shape": (1,)}})
    system = System(derivative=derivative, frame=frame)

    ivp = system.ivp(
        initial={"u": np.array([1.0])},
        interval=Interval(0.0, 0.1, 0.0),
        method=Method(scheme=ExplicitScheme),
        engine=lambda frame: EngineNumpy(frame, accelerator=AcceleratorNone()),
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


def test_system_ivp_final_result_returns_final_working_state_and_step_count() -> None:
    frame = Frame({"u": {"translation": "du", "shape": (1,)}})
    system = System(derivative=derivative, frame=frame)

    ivp = system.ivp(
        initial={"u": np.array([1.0])},
        interval=Interval(0.0, 0.1, 0.3),
        method=Method(scheme=ExplicitScheme),
        engine=lambda frame: EngineNumpy(frame, accelerator=AcceleratorNone()),
        configuration=Configuration(check_progress=False),
    )

    result = ivp.final_result()

    assert abs(result.interval.present - 0.3) < 1.0e-12
    assert result.state is not ivp.initial
    assert result.steps == 3
    np.testing.assert_array_equal(result.state.u, np.array([1.0]))


def test_system_ivp_uses_ready_resolvent_instance() -> None:
    frame = Frame({"u": {"translation": "du", "shape": (1,)}})
    system = System(derivative=derivative, frame=frame)
    resolvent = object()

    ivp = system.ivp(
        initial={"u": np.array([1.0])},
        interval=Interval(0.0, 0.1, 0.0),
        method=Method(scheme=ImplicitScheme, resolvent=resolvent),
        engine=lambda frame: EngineNumpy(frame, accelerator=AcceleratorNone()),
        configuration=Configuration(check_progress=False),
    )

    assert ivp.scheme.resolvent is resolvent
