# tests/schemes/test_scheme_display.py

from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import ImExDerivative
from stark.accelerators import Accelerator
from stark.resolvents import ResolventPicard
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_fixed.euler import SchemeEuler
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.imex_fixed.euler import SchemeIMEXEuler
from stark.schemes.support.display import SchemeDisplay


@dataclass(slots=True)
class DummyTranslation:
    value: float = 0.0

    def __call__(self, origin: object, result: object) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: DummyTranslation) -> DummyTranslation:
        return DummyTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> DummyTranslation:
        return DummyTranslation(scalar * self.value)


class DummyWorkbench:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, dst: object, src: object) -> None:
        del dst, src

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


def dummy_derivative(interval, state, out: DummyTranslation) -> None:
    del interval, state, out


def make_imex_euler() -> SchemeIMEXEuler:
    split = ImExDerivative(
        implicit=dummy_derivative,
        explicit=dummy_derivative,
    )
    workbench = DummyWorkbench()
    resolvent = ResolventPicard(
        split.implicit,
        workbench,
        accelerator=Accelerator.none(),
        tableau=SchemeIMEXEuler.tableau,
    )
    return SchemeIMEXEuler(split, workbench, resolvent=resolvent)


@pytest.mark.parametrize(
    ("scheme", "short_name", "full_name", "tableau_text"),
    [
        (
            SchemeEuler(dummy_derivative, DummyWorkbench()),
            "Euler",
            "Forward Euler",
            "Butcher tableau",
        ),
        (
            SchemeRK4(dummy_derivative, DummyWorkbench()),
            "RK4",
            "Classical Runge-Kutta",
            "Butcher tableau",
        ),
        (
            SchemeBogackiShampine(dummy_derivative, DummyWorkbench()),
            "BS23",
            "Bogacki-Shampine",
            "Butcher tableau",
        ),
        (
            make_imex_euler(),
            "IMEXEuler",
            "IMEX Euler",
            "IMEX Butcher tableau",
        ),
    ],
)
def test_scheme_base_display_output_is_preserved(
    scheme,
    short_name: str,
    full_name: str,
    tableau_text: str,
) -> None:
    rendered_repr = repr(scheme)
    rendered_str = str(scheme)
    rendered_tableau = scheme.display_tableau()

    assert short_name in rendered_repr
    assert full_name in rendered_repr
    assert tableau_text in rendered_repr

    assert short_name in rendered_str
    assert full_name in rendered_str
    assert tableau_text in rendered_str

    assert rendered_str == rendered_tableau
    assert format(scheme, "") == rendered_str


def test_scheme_display_object_matches_descriptor_behaviour() -> None:
    display = SchemeDisplay(SchemeRK4.descriptor, SchemeRK4.tableau)

    assert display.short_name == SchemeRK4.descriptor.short_name
    assert display.full_name == SchemeRK4.descriptor.full_name
    assert display.display_tableau() == SchemeRK4.descriptor.display_tableau(
        SchemeRK4.tableau
    )

    rendered_repr = display.repr_for("SchemeRK4")

    assert "SchemeRK4" in rendered_repr
    assert "RK4" in rendered_repr
    assert "Classical Runge-Kutta" in rendered_repr
    assert "Butcher tableau" in rendered_repr


def test_scheme_display_keeps_repr_exactly_equivalent_to_descriptor() -> None:
    scheme = SchemeRK4(dummy_derivative, DummyWorkbench())

    assert repr(scheme) == SchemeRK4.descriptor.repr_for(
        "SchemeRK4",
        SchemeRK4.tableau,
    )


def test_scheme_display_keeps_tableau_exactly_equivalent_to_descriptor() -> None:
    assert SchemeRK4.display_tableau() == SchemeRK4.descriptor.display_tableau(
        SchemeRK4.tableau
    )