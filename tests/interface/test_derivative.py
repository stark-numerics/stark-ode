from dataclasses import dataclass

import numpy as np
import pytest

from stark.carriers import CarrierNative, CarrierNumpy
from stark.conventions import ConventionInPlace, ConventionReturn
from stark.interface import StarkDerivative, StarkVector
from stark.interface.vector import StarkVectorTranslation
from stark.interface.derivative import (
    BoundInPlaceStarkDerivative,
    BoundReturnStarkDerivative,
)


@dataclass
class DummyInterval:
    present: float


def test_explicit_stark_derivative_construction():
    def rhs(t, y):
        return y

    derivative = StarkDerivative(rhs, ConventionReturn())

    assert derivative.function is rhs
    assert isinstance(derivative.convention, ConventionReturn)


def test_stark_derivative_returning_decorator():
    @StarkDerivative.returning
    def rhs(t, y):
        return y

    assert isinstance(rhs, StarkDerivative)
    assert isinstance(rhs.convention, ConventionReturn)


def test_stark_derivative_in_place_decorator():
    @StarkDerivative.in_place
    def rhs(t, y, dy):
        dy[0] = y[0]

    assert isinstance(rhs, StarkDerivative)
    assert isinstance(rhs.convention, ConventionInPlace)


def test_stark_derivative_from_callable_uses_return_convention():
    def rhs(t, y):
        return y

    derivative = StarkDerivative.from_callable(rhs)

    assert derivative.function is rhs
    assert isinstance(derivative.convention, ConventionReturn)


def test_native_return_derivative_updates_out_value():
    carrier = CarrierNative().bind([1.0, 2.0])

    @StarkDerivative.returning
    def rhs(t, y):
        return [t * y[0], t * y[1]]

    bound = rhs.bind(carrier)

    interval = DummyInterval(present=2.0)
    state = StarkVector([3.0, 4.0], carrier)
    out = StarkVectorTranslation([0.0, 0.0], carrier)

    returned = bound(interval, state, out)

    assert returned is None
    assert out.value == [6.0, 8.0]


def test_numpy_return_derivative_updates_out_value():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))

    @StarkDerivative.returning
    def rhs(t, y):
        return t * y

    bound = rhs.bind(carrier)

    interval = DummyInterval(present=2.0)
    state = StarkVector(np.array([3.0, 4.0]), carrier)
    out = StarkVectorTranslation(np.zeros(2), carrier)

    bound(interval, state, out)

    np.testing.assert_allclose(out.value, np.array([6.0, 8.0]))


def test_numpy_in_place_derivative_updates_out_value():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))

    @StarkDerivative.in_place
    def rhs(t, y, dy):
        dy[:] = t * y

    bound = rhs.bind(carrier)

    interval = DummyInterval(present=2.0)
    state = StarkVector(np.array([3.0, 4.0]), carrier)
    original_out_value = np.zeros(2)
    out = StarkVectorTranslation(original_out_value, carrier)

    bound(interval, state, out)

    assert out.value is original_out_value
    np.testing.assert_allclose(out.value, np.array([6.0, 8.0]))


def test_wrong_numpy_shape_raises():
    carrier = CarrierNumpy(strict_shape=True).bind(np.array([1.0, 2.0]))

    @StarkDerivative.returning
    def rhs(t, y):
        return np.array([1.0, 2.0, 3.0])

    bound = rhs.bind(carrier)

    interval = DummyInterval(present=0.0)
    state = StarkVector(np.array([1.0, 2.0]), carrier)
    out = StarkVectorTranslation(np.zeros(2), carrier)

    with pytest.raises(ValueError):
        bound(interval, state, out)


def test_bound_return_derivative_has_core_compatible_call_shape():
    carrier = CarrierNative().bind(1.0)

    def rhs(t, y):
        return -y

    derivative = StarkDerivative.returning(rhs)
    bound = derivative.bind(carrier)

    interval = DummyInterval(present=0.0)
    state = StarkVector(2.0, carrier)
    out = StarkVectorTranslation(0.0, carrier)

    bound(interval, state, out)

    assert out.value == -2.0


def test_returning_derivative_binds_to_specialized_return_bound_derivative():
    carrier = CarrierNative().bind(1.0)

    def rhs(t, y):
        return -y

    derivative = StarkDerivative.returning(rhs)
    bound = derivative.bind(carrier)

    assert isinstance(bound, BoundReturnStarkDerivative)


def test_in_place_derivative_binds_to_specialized_in_place_bound_derivative():
    carrier = CarrierNative().bind([1.0])

    def rhs(t, y, dy):
        dy[0] = -y[0]

    derivative = StarkDerivative.in_place(rhs)
    bound = derivative.bind(carrier)

    assert isinstance(bound, BoundInPlaceStarkDerivative)
