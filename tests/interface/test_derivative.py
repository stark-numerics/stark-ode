from dataclasses import dataclass

import pytest

from stark.accelerators import AcceleratorNone
from stark.interface import StarkDerivative, StarkDerivativeStyle
from stark.interface.derivative import (
    StarkDerivativeIntervalInPlace,
    StarkDerivativeKernel,
    StarkDerivativeTimeInPlace,
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

    derivative = StarkDerivative(rhs)
    out = DummyTranslation()

    assert isinstance(derivative.implementation, StarkDerivativeTimeInPlace)

    derivative(DummyInterval(2.0), DummyState(3.0), out)

    assert out.dy == 6.0


def test_interval_in_place_signature_passes_interval_object() -> None:
    def rhs(interval, state, out) -> None:
        out.dy = interval.present + state.y

    derivative = StarkDerivative(StarkDerivativeStyle.interval_in_place(rhs))
    out = DummyTranslation()

    assert isinstance(derivative.implementation, StarkDerivativeIntervalInPlace)

    derivative(DummyInterval(2.0), DummyState(3.0), out)

    assert out.dy == 5.0


def test_kernel_signature_calls_field_level_function() -> None:
    calls = []

    def kernel(y, dy, scale) -> None:
        calls.append((y, scale))
        dy[0] = scale * y[0]

    derivative = StarkDerivative(
        StarkDerivativeStyle.kernel(
            kernel,
            state=("y",),
            translation=("dy",),
            parameters=(2.0,),
        )
    )
    state = DummyState([3.0])
    out = DummyTranslation([0.0])

    assert isinstance(derivative.implementation, StarkDerivativeKernel)

    derivative(DummyInterval(0.0), state, out)

    assert calls == [([3.0], 2.0)]
    assert out.dy == [6.0]


def test_kernel_signature_can_be_declared_as_decorator() -> None:
    @StarkDerivativeStyle.kernel(state=("y",), translation=("dy",))
    def kernel(y, dy, scale) -> None:
        dy[0] = scale * y[0]

    derivative = StarkDerivative(kernel.with_parameters(3.0))
    state = DummyState([4.0])
    out = DummyTranslation([0.0])

    assert isinstance(derivative.implementation, StarkDerivativeKernel)

    derivative(DummyInterval(0.0), state, out)

    assert out.dy == [12.0]


def test_kernel_derivative_can_be_accelerated() -> None:
    def kernel(y, dy) -> None:
        dy[0] = y[0]

    derivative = StarkDerivative(
        StarkDerivativeStyle.kernel(kernel, state=("y",), translation=("dy",))
    )

    accelerated = derivative.accelerate(AcceleratorNone())

    assert isinstance(accelerated, StarkDerivativeKernel)


def test_two_argument_callable_requires_explicit_signature() -> None:
    def rhs(t, state):
        return t * state.y

    with pytest.raises(TypeError, match="function\\(t, state, out\\)"):
        StarkDerivative(rhs)
