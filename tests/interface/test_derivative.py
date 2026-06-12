from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark.engines.accelerators import AcceleratorNone
from stark.interface import Derivative, DerivativeStyle, Layout
from stark.interface.derivative import (
    DerivativeAdapterAcceptsInterval,
    DerivativeAdapterReturnsInstant,
    DerivativeKernel,
    DerivativeKernelReturning,
    DerivativeAdapterAcceptsInstant,
)


@dataclass(slots=True)
class DummyInterval:
    present: float


@dataclass(slots=True)
class DummyState:
    y: float


@dataclass(slots=True)
class DummyTranslation:
    dy: float = 0.0


def test_plain_callable_is_recognised_as_time_in_place() -> None:
    def rhs(t, state, out) -> None:
        out.dy = t * state.y

    derivative = Derivative(rhs)
    out = DummyTranslation()

    assert isinstance(derivative.implementation, DerivativeAdapterAcceptsInstant)

    derivative(DummyInterval(2.0), DummyState(3.0), out)

    assert out.dy == 6.0


def test_plain_two_argument_callable_is_recognised_as_returning() -> None:
    def rhs(t, state):
        return t * state.y

    derivative = Derivative(rhs)
    out = DummyTranslation()

    assert isinstance(derivative.implementation, DerivativeAdapterReturnsInstant)

    derivative(DummyInterval(2.0), DummyState(3.0), out)

    assert out.dy == 6.0


def test_returning_signature_can_be_declared_as_decorator() -> None:
    @DerivativeStyle.returning
    def rhs(t, state):
        return {"dy": t * state.y}

    derivative = Derivative(rhs)
    out = SimpleNamespace(
        dy=0.0,
        algebraist_layout=Layout({"y": {"translation": "dy"}}).to_algebraist_layout(),
    )

    derivative(DummyInterval(2.0), DummyState(3.0), out)

    assert out.dy == 6.0


def test_interval_in_place_signature_passes_interval_object() -> None:
    def rhs(interval, state, out) -> None:
        out.dy = interval.present + state.y

    derivative = Derivative(DerivativeStyle.interval_in_place(rhs))
    out = DummyTranslation()

    assert isinstance(derivative.implementation, DerivativeAdapterAcceptsInterval)

    derivative(DummyInterval(2.0), DummyState(3.0), out)

    assert out.dy == 5.0


def test_kernel_signature_calls_field_level_function() -> None:
    calls = []

    def kernel(y, dy, scale) -> None:
        calls.append((y, scale))
        dy[0] = scale * y[0]

    derivative = Derivative(
        DerivativeStyle.kernel(
            kernel,
            state=("y",),
            translation=("dy",),
            parameters=(2.0,),
        )
    )
    state = DummyState([3.0])
    out = DummyTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernel)

    derivative(DummyInterval(0.0), state, out)

    assert calls == [([3.0], 2.0)]
    assert out.dy == [6.0]


def test_kernel_signature_can_be_declared_as_decorator() -> None:
    @DerivativeStyle.kernel(state=("y",), translation=("dy",))
    def kernel(y, dy, scale) -> None:
        dy[0] = scale * y[0]

    derivative = Derivative(kernel.with_parameters(3.0))
    state = DummyState([4.0])
    out = DummyTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernel)

    derivative(DummyInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_returning_kernel_signature_writes_return_value() -> None:
    @DerivativeStyle.kernel_returning(state=("y",), translation=("dy",))
    def kernel(y, scale):
        return [scale * y[0]]

    derivative = Derivative(kernel.with_parameters(3.0))
    state = DummyState([4.0])
    out = DummyTranslation([0.0])

    assert isinstance(derivative.implementation, DerivativeKernelReturning)

    derivative(DummyInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_kernel_derivative_can_be_accelerated() -> None:
    def kernel(y, dy) -> None:
        dy[0] = y[0]

    derivative = Derivative(
        DerivativeStyle.kernel(kernel, state=("y",), translation=("dy",))
    )

    accelerated = derivative.accelerate(AcceleratorNone())

    assert isinstance(accelerated, DerivativeKernel)


def test_returning_kernel_derivative_can_be_accelerated() -> None:
    def kernel(y):
        return y

    derivative = Derivative(
        DerivativeStyle.kernel_returning(kernel, state=("y",), translation=("dy",))
    )

    accelerated = derivative.accelerate(AcceleratorNone())

    assert isinstance(accelerated, DerivativeKernelReturning)


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
        layout=Layout({"y": {"translation": "dy", "shape": (1,)}}),
    )
    ivp = system.ivp(
        initial={"y": jnp.array([2.0])},
        interval=Interval(present=0.0, step=0.1, stop=0.2),
        method=Method(scheme=SchemeEuler),
        engine=lambda layout: EngineJax(layout, dtype=jnp.float32),
        configuration=Configuration(check_progress=False),
    )

    result = ivp.final_result()

    assert float(result.state.y[0]) == pytest.approx(1.805)


def test_unsupported_plain_callable_signature_is_rejected() -> None:
    def rhs(t):
        return t

    with pytest.raises(TypeError, match="function\\(t, state\\)"):
        Derivative(rhs)
