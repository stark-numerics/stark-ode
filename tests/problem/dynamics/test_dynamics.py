from types import SimpleNamespace

import pytest

from stark.engines.accelerators import AcceleratorNone
from stark.problem import Dynamics, DynamicsStyle, Frame
from stark.problem.dynamics import (
    DynamicsAdapterAcceptsIntervalWrites,
    DynamicsAdapterAcceptsInstantReturns,
    DynamicsKernelAcceptsInstantWrites,
    DynamicsKernelAcceptsIntervalWrites,
    DynamicsKernelAcceptsInstantReturns,
    DynamicsKernelAcceptsIntervalReturns,
    DynamicsAdapterAcceptsInstantWrites,
)
from tests.support import (
    DummyDynamicsInterval,
    DummyDynamicsState,
    DummyDynamicsTranslation,
)


def test_plain_callable_is_recognised_as_time_in_place() -> None:
    def rhs(t, state, out) -> None:
        out.dy = t * state.y

    dynamics = Dynamics(rhs)
    out = DummyDynamicsTranslation()

    assert isinstance(dynamics.implementation, DynamicsAdapterAcceptsInstantWrites)

    dynamics(DummyDynamicsInterval(2.0), DummyDynamicsState(3.0), out)

    assert out.dy == 6.0


def test_plain_two_argument_callable_is_recognised_as_returning() -> None:
    def rhs(t, state):
        return t * state.y

    dynamics = Dynamics(rhs)
    out = DummyDynamicsTranslation()

    assert isinstance(dynamics.implementation, DynamicsAdapterAcceptsInstantReturns)

    dynamics(DummyDynamicsInterval(2.0), DummyDynamicsState(3.0), out)

    assert out.dy == 6.0


def test_returning_signature_can_be_declared_as_decorator() -> None:
    @DynamicsStyle.accepts_instant_returns
    def rhs(t, state):
        return {"dy": t * state.y}

    dynamics = Dynamics(rhs)
    out = SimpleNamespace(
        dy=0.0,
        frame=Frame({"y": {"translation": "dy"}}),
    )

    dynamics(DummyDynamicsInterval(2.0), DummyDynamicsState(3.0), out)

    assert out.dy == 6.0


def test_interval_in_place_signature_passes_interval_object() -> None:
    def rhs(interval, state, out) -> None:
        out.dy = interval.present + state.y

    dynamics = Dynamics(DynamicsStyle.accepts_interval_writes(rhs))
    out = DummyDynamicsTranslation()

    assert isinstance(dynamics.implementation, DynamicsAdapterAcceptsIntervalWrites)

    dynamics(DummyDynamicsInterval(2.0), DummyDynamicsState(3.0), out)

    assert out.dy == 5.0


def test_kernel_signature_calls_field_level_function() -> None:
    calls = []

    def kernel(t, y, dy, scale) -> None:
        calls.append((t, y, scale))
        dy[0] = scale * y[0]

    dynamics = Dynamics(
        DynamicsStyle.kernel_accepts_instant_writes(
            kernel,
            state=("y",),
            translation=("dy",),
            parameters=(2.0,),
        )
    )
    state = DummyDynamicsState([3.0])
    out = DummyDynamicsTranslation([0.0])

    assert isinstance(dynamics.implementation, DynamicsKernelAcceptsInstantWrites)

    dynamics(DummyDynamicsInterval(0.0), state, out)

    assert calls == [(0.0, [3.0], 2.0)]
    assert out.dy == [6.0]


def test_kernel_signature_can_be_declared_as_decorator() -> None:
    @DynamicsStyle.kernel_accepts_instant_writes(state=("y",), translation=("dy",))
    def kernel(t, y, dy, scale) -> None:
        dy[0] = scale * y[0]

    dynamics = Dynamics(kernel.with_parameters(3.0))
    state = DummyDynamicsState([4.0])
    out = DummyDynamicsTranslation([0.0])

    assert isinstance(dynamics.implementation, DynamicsKernelAcceptsInstantWrites)

    dynamics(DummyDynamicsInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_interval_kernel_signature_passes_interval_object() -> None:
    @DynamicsStyle.kernel_accepts_interval_writes(state=("y",), translation=("dy",))
    def kernel(interval, y, dy, scale) -> None:
        t = interval.present
        dy[0] = scale * t * y[0]

    dynamics = Dynamics(kernel.with_parameters(3.0))
    state = DummyDynamicsState([4.0])
    out = DummyDynamicsTranslation([0.0])

    assert isinstance(dynamics.implementation, DynamicsKernelAcceptsIntervalWrites)

    dynamics(DummyDynamicsInterval(2.0), state, out)

    assert out.dy == [24.0]


def test_returning_kernel_signature_writes_return_value() -> None:
    @DynamicsStyle.kernel_accepts_instant_returns(state=("y",), translation=("dy",))
    def kernel(t, y, scale):
        return [scale * y[0]]

    dynamics = Dynamics(kernel.with_parameters(3.0))
    state = DummyDynamicsState([4.0])
    out = DummyDynamicsTranslation([0.0])

    assert isinstance(dynamics.implementation, DynamicsKernelAcceptsInstantReturns)

    dynamics(DummyDynamicsInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_interval_returning_kernel_signature_passes_interval_object() -> None:
    @DynamicsStyle.kernel_accepts_interval_returns(state=("y",), translation=("dy",))
    def kernel(interval, y, scale):
        t = interval.present
        return [scale * t * y[0]]

    dynamics = Dynamics(kernel.with_parameters(3.0))
    state = DummyDynamicsState([4.0])
    out = DummyDynamicsTranslation([0.0])

    assert isinstance(dynamics.implementation, DynamicsKernelAcceptsIntervalReturns)

    dynamics(DummyDynamicsInterval(2.0), state, out)

    assert out.dy == [24.0]


def test_kernel_dynamics_can_be_accelerated() -> None:
    def kernel(t, y, dy) -> None:
        dy[0] = y[0]

    dynamics = Dynamics(
        DynamicsStyle.kernel_accepts_instant_writes(kernel, state=("y",), translation=("dy",))
    )

    accelerated = dynamics.accelerate(AcceleratorNone())

    assert isinstance(accelerated, DynamicsKernelAcceptsInstantWrites)


def test_returning_kernel_dynamics_can_be_accelerated() -> None:
    def kernel(t, y):
        return y

    dynamics = Dynamics(
        DynamicsStyle.kernel_accepts_instant_returns(kernel, state=("y",), translation=("dy",))
    )

    accelerated = dynamics.accelerate(AcceleratorNone())

    assert isinstance(accelerated, DynamicsKernelAcceptsInstantReturns)


def test_returning_dynamics_runs_through_jax_engine() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from stark import Configuration, Interval, Method, System
    from stark.engines import EngineJax
    from stark.methods.schemes import SchemeEuler

    def rhs(t, state):
        del t
        return -0.5 * state.y

    system = System(
        dynamics=rhs,
        frame=Frame({"y": {"translation": "dy", "shape": (1,)}}),
    )
    ivp = system.ivp(
        initial={"y": jnp.array([2.0])},
        interval=Interval(present=0.0, step=0.1, stop=0.2),
        method=Method(scheme=SchemeEuler),
        engine=lambda frame: EngineJax(frame, dtype=jnp.float32),
        configuration=Configuration(check_progress=False),
    )

    result = ivp.final_result()

    assert float(result.state.y[0]) == pytest.approx(1.805)


def test_unsupported_plain_callable_signature_is_rejected() -> None:
    def rhs(t):
        return t

    with pytest.raises(TypeError, match="function\\(t, state\\)"):
        Dynamics(rhs)
