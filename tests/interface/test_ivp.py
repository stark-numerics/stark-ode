from dataclasses import dataclass

import numpy as np
import pytest

from stark import Executor, Integrator, Interval, Marcher
from stark.carriers import CarrierError, CarrierLibrary, CarrierNative
from stark.interface import StarkDerivative, StarkVector
from stark.interface.derivative import BoundReturnStarkDerivative
from stark.interface.ivp import StarkIVP, StarkIVPBuild
from stark.interface.vector import StarkVectorTranslation, StarkVectorWorkbench
from stark.routing import Routing, RoutingVectorReturn
from stark.schemes import SchemeCashKarp, SchemeEuler


@dataclass
class DummyInterval:
    present: float
    step: float
    stop: float

    def copy(self) -> "DummyInterval":
        return DummyInterval(
            present=self.present,
            step=self.step,
            stop=self.stop,
        )

    def increment(self, dt: float) -> None:
        self.present += dt


def test_can_construct_stark_ivp_with_derivative_initial_and_interval():
    def rhs(t, y):
        return -y

    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=rhs,
        initial=1.0,
        interval=interval,
    )

    assert ivp.derivative is rhs
    assert ivp.initial == 1.0
    assert ivp.interval is interval
    assert isinstance(ivp.carrier_library, CarrierLibrary)


def test_can_construct_with_explicit_carrier():
    carrier = CarrierNative()
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        carrier=carrier,
    )

    assert ivp.carrier is carrier
    assert ivp.carrier_library is None


def test_can_construct_with_explicit_carrier_library():
    carrier_library = CarrierLibrary((CarrierNative(),))
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        carrier_library=carrier_library,
    )

    assert ivp.carrier_library is carrier_library


def test_can_construct_with_explicit_routing():
    routing = Routing(vector=RoutingVectorReturn())
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        routing=routing,
    )

    assert ivp.routing is routing


def test_can_construct_with_explicit_executor():
    executor = object()
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        executor=executor,
    )

    assert ivp.executor is executor


def test_can_construct_with_explicit_scheme():
    scheme = object()
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        scheme=scheme,
    )

    assert ivp.scheme is scheme


def test_raw_scalar_initial_becomes_stark_vector():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1,
        interval=interval,
    )

    assert isinstance(ivp.prepared_initial, StarkVector)
    assert ivp.prepared_initial.value == 1.0
    assert ivp.prepared_initial.carrier is ivp.prepared_carrier


def test_raw_list_initial_becomes_stark_vector():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: [-value for value in y],
        initial=[1, 2],
        interval=interval,
    )

    assert isinstance(ivp.prepared_initial, StarkVector)
    assert ivp.prepared_initial.value == [1.0, 2.0]
    assert ivp.prepared_initial.carrier is ivp.prepared_carrier


def test_raw_numpy_initial_becomes_stark_vector():
    initial = np.array([1.0, 2.0])
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=initial,
        interval=interval,
    )

    assert isinstance(ivp.prepared_initial, StarkVector)
    np.testing.assert_allclose(ivp.prepared_initial.value, initial)
    assert ivp.prepared_initial.carrier is ivp.prepared_carrier


def test_explicit_carrier_is_respected_during_initial_preparation():
    carrier = CarrierNative()
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1,
        interval=interval,
        carrier=carrier,
    )

    assert ivp.carrier is carrier
    assert ivp.prepared_initial.value == 1.0
    assert ivp.prepared_carrier.carrier is carrier


def test_carrier_library_is_used_when_carrier_missing():
    native = CarrierNative()
    carrier_library = CarrierLibrary((native,))
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        carrier_library=carrier_library,
    )

    assert ivp.carrier_library is carrier_library
    assert ivp.prepared_carrier.carrier is native


def test_explicit_routing_object_sets_vector_routing():
    routing = Routing(vector=RoutingVectorReturn())
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        routing=routing,
    )

    assert ivp.vector_routing is routing.vector


def test_bare_vector_routing_is_rejected():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    with pytest.raises(TypeError, match="Routing"):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=interval,
            routing=RoutingVectorReturn(),
        )


def test_explicit_carrier_library_is_respected():
    native = CarrierNative()
    carrier_library = CarrierLibrary((native,))
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        carrier_library=carrier_library,
    )

    assert ivp.carrier_library is carrier_library
    assert ivp.prepared_carrier.carrier is native


def test_raw_callable_derivative_is_wrapped_and_bound():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    def rhs(t, y):
        return -y

    ivp = StarkIVP(
        derivative=rhs,
        initial=1.0,
        interval=interval,
    )

    assert isinstance(ivp.prepared_derivative, BoundReturnStarkDerivative)


def test_stark_derivative_is_accepted_and_bound():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    @StarkDerivative.returning
    def rhs(t, y):
        return -y

    ivp = StarkIVP(
        derivative=rhs,
        initial=1.0,
        interval=interval,
    )

    assert isinstance(ivp.prepared_derivative, BoundReturnStarkDerivative)


def test_prepared_derivative_uses_prepared_initial_carrier():
    interval = DummyInterval(present=2.0, step=0.1, stop=1.0)

    def rhs(t, y):
        return t * y

    ivp = StarkIVP(
        derivative=rhs,
        initial=3.0,
        interval=interval,
    )

    state = ivp.prepared_initial
    out = StarkVectorTranslation(
        ivp.prepared_carrier.zero_translation(),
        ivp.prepared_carrier,
        ivp.vector_routing,
    )

    ivp.prepared_derivative(interval, state, out)

    assert out.value == 6.0


def test_prepared_derivative_works_with_numpy_carrier():
    interval = DummyInterval(present=2.0, step=0.1, stop=1.0)
    initial = np.array([3.0, 4.0])

    def rhs(t, y):
        return t * y

    ivp = StarkIVP(
        derivative=rhs,
        initial=initial,
        interval=interval,
    )

    out = StarkVectorTranslation(
        ivp.prepared_carrier.zero_translation(),
        ivp.prepared_carrier,
        ivp.vector_routing,
    )

    ivp.prepared_derivative(interval, ivp.prepared_initial, out)

    np.testing.assert_allclose(out.value, np.array([6.0, 8.0]))


@dataclass
class MissingStepInterval:
    present: float
    stop: float

    def copy(self) -> "MissingStepInterval":
        return MissingStepInterval(present=self.present, stop=self.stop)

    def increment(self, dt: float) -> None:
        self.present += dt


@dataclass
class InvalidInterval:
    present: float
    step: float | None
    stop: float

    def copy(self) -> "InvalidInterval":
        return InvalidInterval(
            present=self.present,
            step=self.step,
            stop=self.stop,
        )

    def increment(self, dt: float) -> None:
        self.present += dt


@dataclass
class MissingCopyInterval:
    present: float
    step: float
    stop: float

    def increment(self, dt: float) -> None:
        self.present += dt


@dataclass
class MissingIncrementInterval:
    present: float
    step: float
    stop: float

    def copy(self) -> "MissingIncrementInterval":
        return MissingIncrementInterval(
            present=self.present,
            step=self.step,
            stop=self.stop,
        )


def test_interval_object_is_accepted():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    assert ivp.interval is interval


def test_tuple_interval_is_rejected():
    with pytest.raises(TypeError):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=(0.0, 1.0),
        )


def test_list_interval_is_rejected():
    with pytest.raises(TypeError):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=[0.0, 1.0],
        )


def test_interval_missing_step_is_rejected():
    interval = MissingStepInterval(present=0.0, stop=1.0)

    with pytest.raises(TypeError):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=interval,
        )


def test_interval_missing_copy_is_rejected():
    interval = MissingCopyInterval(present=0.0, step=0.1, stop=1.0)

    with pytest.raises(TypeError, match="copy"):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=interval,
        )


def test_interval_missing_increment_is_rejected():
    interval = MissingIncrementInterval(present=0.0, step=0.1, stop=1.0)

    with pytest.raises(TypeError, match="increment"):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=interval,
        )


def test_interval_with_none_step_is_rejected():
    interval = InvalidInterval(present=0.0, step=None, stop=1.0)

    with pytest.raises(ValueError):
        StarkIVP(
            derivative=lambda t, y: -y,
            initial=1.0,
            interval=interval,
        )

def test_build_returns_stark_ivp_build():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    build = ivp.build()

    assert isinstance(build, StarkIVPBuild)


def test_build_creates_workbench():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    build = ivp.build()

    assert isinstance(build.workbench, StarkVectorWorkbench)
    assert build.workbench.carrier is ivp.prepared_carrier
    assert build.workbench.routing is ivp.vector_routing


def test_build_exposes_prepared_initial_and_derivative():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    build = ivp.build()

    assert build.initial is ivp.prepared_initial
    assert build.derivative is ivp.prepared_derivative
    assert build.interval is interval


def test_build_creates_default_scheme_executor_marcher_and_integrator():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    build = ivp.build()

    assert isinstance(build.scheme, SchemeCashKarp)
    assert isinstance(build.executor, Executor)
    assert isinstance(build.marcher, Marcher)
    assert isinstance(build.integrator, Integrator)


def test_build_uses_supplied_executor():
    executor = Executor()
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        executor=executor,
    )

    build = ivp.build()

    assert build.executor is executor


class DummyScheme:
    def __init__(self):
        self.apply_delta_safety = None

    def __call__(self, interval, state, executor):
        return interval.step

    def snapshot_state(self, state):
        return state

    def set_apply_delta_safety(self, enabled):
        self.apply_delta_safety = enabled


def test_build_uses_supplied_scheme_instance():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)
    supplied_scheme = DummyScheme()

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        scheme=supplied_scheme,
    )

    build = ivp.build()

    assert build.scheme is supplied_scheme


def test_build_accepts_scheme_class():
    interval = DummyInterval(present=0.0, step=0.1, stop=1.0)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
        scheme=SchemeEuler,
    )

    build = ivp.build()

    assert isinstance(build.scheme, SchemeEuler)


def test_integrate_returns_iterable_result():
    interval = Interval(present=0.0, step=0.1, stop=0.2)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    result = ivp.integrate()

    assert iter(result) is result


def test_integrate_native_scalar_smoke():
    interval = Interval(present=0.0, step=0.1, stop=0.2)

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=1.0,
        interval=interval,
    )

    results = list(ivp.integrate())

    assert results

    final_interval, final_state = results[-1]

    assert final_interval.present == pytest.approx(0.2)
    assert 0.0 < final_state.value < 1.0


def test_integrate_numpy_smoke():
    interval = Interval(present=0.0, step=0.1, stop=0.2)
    initial = np.array([1.0, 2.0])

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=initial,
        interval=interval,
    )

    results = list(ivp.integrate())

    assert results

    final_interval, final_state = results[-1]

    assert final_interval.present == pytest.approx(0.2)
    np.testing.assert_array_less(final_state.value, initial)
    np.testing.assert_array_less(np.zeros_like(initial), final_state.value)


def test_stark_ivp_is_exported_from_interface_package():
    from stark.interface import StarkIVP

    assert StarkIVP is not None
