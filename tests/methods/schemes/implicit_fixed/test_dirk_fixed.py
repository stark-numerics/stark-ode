from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Interval, Tolerance
from stark.core.contracts import IntervalLike
from stark.engines.accelerators import AcceleratorNone
from stark.engines.algebraist.runtime import AlgebraistRuntimeSpecialist
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.implicit.fixed.crouzeix_dirk3 import SchemeCrouzeixDIRK3


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


@dataclass(slots=True)
class ArrayScalarState:
    value: np.ndarray

    @classmethod
    def zero(cls) -> "ArrayScalarState":
        return cls(np.zeros(1))


@dataclass(slots=True)
class ArrayScalarTranslation:
    value: np.ndarray

    @classmethod
    def zero(cls) -> "ArrayScalarTranslation":
        return cls(np.zeros(1))

    def __call__(self, origin: ArrayScalarState, result: ArrayScalarState) -> None:
        result.value[...] = origin.value + self.value

    def norm(self) -> float:
        return float(abs(self.value[0]))

    def __add__(self, other: "ArrayScalarTranslation") -> "ArrayScalarTranslation":
        return ArrayScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ArrayScalarTranslation":
        return ArrayScalarTranslation(scalar * self.value)


class ArrayScalarAllocator:
    def allocate_state(self) -> ArrayScalarState:
        return ArrayScalarState.zero()

    def copy_state(self, source: ArrayScalarState, out: ArrayScalarState) -> None:
        out.value[...] = source.value

    def allocate_translation(self) -> ArrayScalarTranslation:
        return ArrayScalarTranslation.zero()


def constant_rhs(
    interval: IntervalLike,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def array_constant_rhs(
    interval: IntervalLike,
    state: ArrayScalarState,
    out: ArrayScalarTranslation,
) -> None:
    del interval, state
    out.value[...] = 1.0


def make_scheme() -> SchemeCrouzeixDIRK3:
    allocator = ScalarAllocator()
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=SchemeCrouzeixDIRK3.tableau,
    )
    return SchemeCrouzeixDIRK3(
        constant_rhs,
        allocator,
        resolvent=resolvent,
    )


def make_array_scalar_scheme(
    *,
    specialist: bool = False,
) -> SchemeCrouzeixDIRK3:
    allocator = ArrayScalarAllocator()
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=SchemeCrouzeixDIRK3.tableau,
    )
    return SchemeCrouzeixDIRK3(
        array_constant_rhs,
        allocator,
        resolvent=resolvent,
        specialist=(
            AlgebraistRuntimeSpecialist(
                translation=allocator.allocate_translation(),
                allocator=allocator,
            )
            if specialist
            else None
        ),
    )


def test_crouzeix_dirk3_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeCrouzeixDIRK3.__dict__


def test_crouzeix_dirk3_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = make_scheme()

    assert scheme.redirect_call == scheme.call_step


def test_crouzeix_dirk3_specialist_path_is_scheme_owned_generated_call() -> None:
    scheme = make_array_scalar_scheme(specialist=True)

    assert scheme.redirect_call == scheme.call_step


def test_crouzeix_dirk3_public_call_uses_redirect_call() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    def replacement_call(
        interval: IntervalLike,
        state,
    ) -> float:
        del interval
        state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_crouzeix_dirk3_call_performs_one_constant_rhs_step() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.125)
    assert interval.step == pytest.approx(0.125)


def test_crouzeix_dirk3_specialist_path_matches_generic_path() -> None:
    generic = make_array_scalar_scheme()
    generated = make_array_scalar_scheme(specialist=True)
    generic_interval = Interval(present=0.0, step=0.125, stop=1.0)
    generated_interval = Interval(present=0.0, step=0.125, stop=1.0)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state)
    generated_dt = generated(generated_interval, generated_state)

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


def test_crouzeix_dirk3_specialist_kernels_are_prepared() -> None:
    scheme = make_array_scalar_scheme(specialist=True)

    assert callable(scheme.known2_kernel)
    assert callable(scheme.known3_kernel)
    assert callable(scheme.known4_kernel)
    assert callable(scheme.final_update)


def test_crouzeix_dirk3_call_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(0.05)
    assert interval.step == pytest.approx(0.0)


def test_crouzeix_dirk3_returns_zero_when_interval_is_complete() -> None:
    scheme = make_scheme()
    interval = Interval(present=1.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.0)
    assert state.value == pytest.approx(0.0)


def test_crouzeix_dirk3_snapshot_is_exposed_through_scheme() -> None:
    scheme = make_scheme()
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)
