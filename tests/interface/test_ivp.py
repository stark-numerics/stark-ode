import numpy as np
import pytest

from stark import Interval
from stark.carriers import CarrierNative, CarrierNumpy
from stark.interface import StarkDerivative, StarkIVP, StarkVector
from stark.interface.vector import StarkVectorWorkbench


def test_ivp_prepares_native_initial_by_default() -> None:
    ivp = StarkIVP(
        derivative=lambda t, y: y,
        initial=[1.0, 2.0],
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    assert isinstance(ivp.prepared_initial, StarkVector)
    assert isinstance(ivp.prepared_carrier, CarrierNative)
    assert ivp.prepared_initial.carrier is ivp.prepared_carrier
    assert ivp.prepared_initial.value == [1.0, 2.0]


def test_ivp_prepares_numpy_initial_by_default() -> None:
    initial = np.array([1.0, 2.0])

    ivp = StarkIVP(
        derivative=lambda t, y: y,
        initial=initial,
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    assert isinstance(ivp.prepared_carrier, CarrierNumpy)
    assert ivp.prepared_initial.carrier is ivp.prepared_carrier
    np.testing.assert_allclose(ivp.prepared_initial.value, initial)


def test_ivp_accepts_explicit_carrier() -> None:
    initial = np.array([1.0, 2.0])
    carrier = CarrierNumpy(initial)

    ivp = StarkIVP(
        derivative=lambda t, y: y,
        initial=initial,
        carrier=carrier,
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    assert ivp.prepared_carrier is carrier
    assert ivp.prepared_initial.carrier is carrier


def test_ivp_accepts_prepared_stark_vector() -> None:
    carrier = CarrierNative([1.0, 2.0])
    initial = StarkVector([1.0, 2.0], carrier)

    ivp = StarkIVP(
        derivative=lambda t, y: y,
        initial=initial,
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    assert ivp.prepared_initial is initial
    assert ivp.prepared_carrier is carrier


def test_ivp_rejects_explicit_carrier_with_prepared_stark_vector() -> None:
    carrier = CarrierNative([1.0, 2.0])
    initial = StarkVector([1.0, 2.0], carrier)

    with pytest.raises(TypeError, match="explicit carrier"):
        StarkIVP(
            derivative=lambda t, y: y,
            initial=initial,
            carrier=carrier,
            interval=Interval(present=0.0, step=0.1, stop=0.2),
        )


def test_ivp_rejects_tuple_interval() -> None:
    with pytest.raises(TypeError, match="explicit interval-like object"):
        StarkIVP(
            derivative=lambda t, y: y,
            initial=[1.0, 2.0],
            interval=(0.0, 1.0),
        )


def test_ivp_rejects_interval_missing_required_attributes() -> None:
    class BadInterval:
        present = 0.0

        def copy(self):
            return self

        def increment(self) -> None:
            pass

    with pytest.raises(TypeError, match="Missing"):
        StarkIVP(
            derivative=lambda t, y: y,
            initial=[1.0, 2.0],
            interval=BadInterval(),
        )


def test_ivp_rejects_interval_with_none_attributes() -> None:
    class BadInterval:
        present = 0.0
        step = None
        stop = 1.0

        def copy(self):
            return self

        def increment(self) -> None:
            pass

    with pytest.raises(ValueError, match="step"):
        StarkIVP(
            derivative=lambda t, y: y,
            initial=[1.0, 2.0],
            interval=BadInterval(),
        )


def test_ivp_prepares_returning_derivative() -> None:
    ivp = StarkIVP(
        derivative=StarkDerivative.returning(lambda t, y: [t * item for item in y]),
        initial=[1.0, 2.0],
        interval=Interval(present=2.0, step=0.1, stop=0.2),
    )

    out = ivp.prepared_initial.zero_translation()

    ivp.prepared_derivative(ivp.interval, ivp.prepared_initial, out)

    assert out.value == [2.0, 4.0]


def test_ivp_prepares_in_place_derivative() -> None:
    def rhs(t, y, dy) -> None:
        dy[0] = t * y[0]
        dy[1] = t * y[1]

    ivp = StarkIVP(
        derivative=StarkDerivative.in_place(rhs),
        initial=np.array([1.0, 2.0]),
        interval=Interval(present=2.0, step=0.1, stop=0.2),
    )

    out = ivp.prepared_initial.zero_translation()

    ivp.prepared_derivative(ivp.interval, ivp.prepared_initial, out)

    np.testing.assert_allclose(out.value, [2.0, 4.0])


def test_ivp_build_creates_runtime_components() -> None:
    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=np.array([1.0, 2.0]),
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    runtime = ivp.build()

    assert isinstance(runtime.workbench, StarkVectorWorkbench)
    assert runtime.workbench.carrier is ivp.prepared_carrier
    assert runtime.derivative is ivp.prepared_derivative
    assert runtime.initial is ivp.prepared_initial
    assert runtime.interval is ivp.interval


def test_ivp_integrate_numpy_smoke() -> None:
    initial = np.array([1.0, 2.0])

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=initial,
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    results = list(ivp.integrate())

    assert results
    final_interval, final_state = results[-1]

    assert final_interval.present == pytest.approx(0.2)
    assert np.all(final_state.value > 0.0)
    assert np.all(final_state.value < initial)