from __future__ import annotations

from dataclasses import dataclass

from stark.core.auditor import Auditor
from stark.contracts import Derivative, State, Translation, Allocator
from stark.schemes.execution.step_support import SchemeStepSupport


@dataclass(slots=True)
class SchemeSupportExplicit:
    """Reusable setup support for explicit schemes.

    This object owns the non-algorithmic setup that explicit schemes share:
    derivative binding, input audit, step-support construction, the shared
    first translation buffer, and snapshots.

    Concrete schemes should still own their actual step algorithm.
    """

    derivative: Derivative
    step_support: SchemeStepSupport
    first_translation: Translation

    @classmethod
    def from_inputs(
        cls,
        derivative: Derivative,
        allocator: Allocator,
    ) -> SchemeSupportExplicit:
        first_translation = allocator.allocate_translation()
        Auditor.require_scheme_inputs(derivative, allocator, first_translation)

        return cls(
            derivative=derivative,
            step_support=SchemeStepSupport(allocator, first_translation),
            first_translation=first_translation,
        )

    @property
    def k1(self) -> Translation:
        return self.first_translation

    def snapshot_state(self, state: State) -> State:
        return self.step_support.snapshot_state(state)


def initialise_explicit_support(
    scheme,
    derivative: Derivative,
    allocator: Allocator,
) -> SchemeSupportExplicit:
    """Attach the standard explicit scheme support object to a scheme."""

    support = SchemeSupportExplicit.from_inputs(derivative, allocator)
    scheme.explicit = support
    scheme.derivative = support.derivative
    scheme.workspace = support.step_support
    scheme.k1 = support.k1
    return support


def explicit_snapshot_state(self, state: State) -> State:
    """Return an explicit scheme-owned snapshot of *state*."""

    return self.explicit.snapshot_state(state)


__all__ = [
    "SchemeSupportExplicit",
    "explicit_snapshot_state",
    "initialise_explicit_support",
]
