from types import SimpleNamespace

import pytest

from stark.engines.shared.accelerators import AcceleratorNone
from stark.problem import Derivative, DerivativeStyle, Frame
from stark.problem.derivative import (
    DerivativeAdapterAcceptsIntervalWrites,
    DerivativeAdapterAcceptsInstantReturns,
    DerivativeKernelAcceptsInstantWrites,
    DerivativeKernelAcceptsIntervalWrites,
    DerivativeKernelAcceptsInstantReturns,
    DerivativeKernelAcceptsIntervalReturns,
    DerivativeAdapterAcceptsInstantWrites,
)
from tests.support import (
    DummyDerivativeInterval,
    DummyDerivativeState,
    DummyDerivativeTranslation,
)


def test_plain_callable_is_recognised_as_time_in_place() -> None:
    def rhs(t, state, out) -> None:
        out.dy = t * state.y

    derivative = Derivative(rhs)
    out = DummyDerivativeTranslation()

    assert isinstance(derivative.implementation, DerivativeAdapterAcceptsInstantWrites)

    derivative(DummyDerivativeInterval(2.0), DummyDerivativeState(3.0), out)

    assert out.dy == 6.0


def test_plain_two_argument_callable_is_recognised_as_returning() -> None:
    def rhs(t, state):
        return t * state.y

    derivative = Derivative(rhs)
    out = DummyDerivativeTranslation()

    assert isinstance(derivative.implementation, DerivativeAdapterAcceptsInstantReturns)

    derivative(DummyDerivativeInterval(2.0), DummyDerivativeState(3.0), out)

    assert out.dy == 6.0


def test_returning_signature_can_be_declared_as_decorator() -> None:
    @DerivativeStyle.accepts_instant_returns
    def rhs(t, state):
        return {"dy": t * state.y}

    derivative = Derivative(rhs)
    out = SimpleNamespace(
        dy=0.0,
        algebraist_frame=Frame({"y": {"translation": "dy"}}).to_algebraist_frame(),
    )

    derivative(DummyDerivativeInterval(2.0), DummyDerivativeState(3.0), out)

    assert out.dy == 6.0


def test_interval_in_place_signature_passes_interval_object() -> None:
    def rhs(interval, state, out) -> None:
        out.dy = interval.present + state.y

    derivative = Derivative(DerivativeStyle.accepts_interval_writes(rhs))
    out = DummyDerivativeTranslation()

    assert isinstance(derivative.implementation, DerivativeAdapterAcceptsIntervalWrites)

    derivative(DummyDerivativeInterval(2.0), DummyDerivativeState(3.0), out)

    assert out.dy == 5.0


def test_kernel_signature_calls_field_level_function() -> None:
    calls = []

    def kernel(t, y, dy, scale) -> None:
        calls.append((t, y, scale))
        dy[0] = scale * y[0]

    derivative = Derivative(
        DerivativeStyle.kernel_accepts_instant_writes(
            kernel,
            state=("y",),
            translation=("dy",),
            parameters=(2.0,),
        )
    )
    state = DummyDerivativeState([3.0])
    out = DummyDerivativeTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernelAcceptsInstantWrites)

    derivative(DummyDerivativeInterval(0.0), state, out)

    assert calls == [(0.0, [3.0], 2.0)]
    assert out.dy == [6.0]


def test_kernel_signature_can_be_declared_as_decorator() -> None:
    @DerivativeStyle.kernel_accepts_instant_writes(state=("y",), translation=("dy",))
    def kernel(t, y, dy, scale) -> None:
        dy[0] = scale * y[0]

    derivative = Derivative(kernel.with_parameters(3.0))
    state = DummyDerivativeState([4.0])
    out = DummyDerivativeTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernelAcceptsInstantWrites)

    derivative(DummyDerivativeInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_interval_kernel_signature_passes_interval_object() -> None:
    @DerivativeStyle.kernel_accepts_interval_writes(state=("y",), translation=("dy",))
    def kernel(interval, y, dy, scale) -> None:
        t = interval.present
        dy[0] = scale * t * y[0]

    derivative = Derivative(kernel.with_parameters(3.0))
    state = DummyDerivativeState([4.0])
    out = DummyDerivativeTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernelAcceptsIntervalWrites)

    derivative(DummyDerivativeInterval(2.0), state, out)

    assert out.dy == [24.0]


def test_returning_kernel_signature_writes_return_value() -> None:
    @DerivativeStyle.kernel_accepts_instant_returns(state=("y",), translation=("dy",))
    def kernel(t, y, scale):
        return [scale * y[0]]

    derivative = Derivative(kernel.with_parameters(3.0))
    state = DummyDerivativeState([4.0])
    out = DummyDerivativeTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernelAcceptsInstantReturns)

    derivative(DummyDerivativeInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_interval_returning_kernel_signature_passes_interval_object() -> None:
    @DerivativeStyle.kernel_accepts_interval_returns(state=("y",), translation=("dy",))
    def kernel(interval, y, scale):
        t = interval.present
        return [scale * t * y[0]]

    derivative = Derivative(kernel.with_parameters(3.0))
    state = DummyDerivativeState([4.0])
    out = DummyDerivativeTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernelAcceptsIntervalReturns)

    derivative(DummyDerivativeInterval(2.0), state, out)

    assert out.dy == [24.0]


def test_kernel_derivative_can_be_accelerated() -> None:
    def kernel(t, y, dy) -> None:
        dy[0] = y[0]

    derivative = Derivative(
        DerivativeStyle.kernel_accepts_instant_writes(kernel, state=("y",), translation=("dy",))
    )

    accelerated = derivative.accelerate(AcceleratorNone())

    assert isinstance(accelerated, DerivativeKernelAcceptsInstantWrites)


def test_returning_kernel_derivative_can_be_accelerated() -> None:
    def kernel(t, y):
        return y

    derivative = Derivative(
        DerivativeStyle.kernel_accepts_instant_returns(kernel, state=("y",), translation=("dy",))
    )

    accelerated = derivative.accelerate(AcceleratorNone())

    assert isinstance(accelerated, DerivativeKernelAcceptsInstantReturns)


def test_returning_derivative_runs_through_jax_engine() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from stark import Configuration, Interval, Method, System
    from stark.engines import EngineJax
    from stark.methods.schemes import SchemeEuler

    def rhs(t, state):
        del t
        return -0.5 * state.y

    system = System(
        derivative=rhs,
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
        Derivative(rhs)
