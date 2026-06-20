# tests/schemes/test_with_scheme_display.py

from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Derivative
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents import ResolventPicard
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from stark.methods.schemes.imex.fixed.euler import SchemeIMEXEuler
from stark.methods.schemes.display.display import SchemeDisplay


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


class DummyAllocator:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, source: object, out: object) -> None:
        del out, source

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


def dummy_derivative(interval, state, out: DummyTranslation) -> None:
    del interval, state, out


def make_imex_euler() -> SchemeIMEXEuler:
    split = Derivative.imex(
        implicit=dummy_derivative,
        explicit=dummy_derivative,
    )
    allocator = DummyAllocator()
    resolvent = ResolventPicard(
        allocator,
        accelerator=AcceleratorNone(),
        tableau=SchemeIMEXEuler.tableau,
    )
    return SchemeIMEXEuler(split, allocator, resolvent=resolvent)


@pytest.mark.parametrize(
    ("scheme", "short_name", "full_name", "tableau_text"),
    [
        (
            SchemeEuler(dummy_derivative, DummyAllocator()),
            "Euler",
            "Forward Euler",
            "Butcher tableau",
        ),
        (
            SchemeRK4(dummy_derivative, DummyAllocator()),
            "RK4",
            "Classical Runge-Kutta",
            "Butcher tableau",
        ),
        (
            SchemeBogackiShampine(dummy_derivative, DummyAllocator()),
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


def test_with_scheme_display_object_matches_descriptor_behaviour() -> None:
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


def test_with_scheme_display_keeps_repr_exactly_equivalent_to_descriptor() -> None:
    scheme = SchemeRK4(dummy_derivative, DummyAllocator())

    assert repr(scheme) == SchemeRK4.descriptor.repr_for(
        "SchemeRK4",
        SchemeRK4.tableau,
    )


def test_with_scheme_display_keeps_tableau_exactly_equivalent_to_descriptor() -> None:
    assert SchemeRK4.display_tableau() == SchemeRK4.descriptor.display_tableau(
        SchemeRK4.tableau
    )
