from __future__ import annotations

from typing import Protocol

from stark.core.block import BlockAllocator
from stark.core.contracts import DynamicsLike, State, Allocator, Resolvent
from stark.methods.schemes.execution.step_support import SchemeStepSupport
from stark.methods.schemes.method.descriptor import SchemeDescriptor


class SchemeRuntimeImplicitOwner(Protocol):
    descriptor: SchemeDescriptor
    resolvent: Resolvent


class SchemeRuntimeImplicit:
    """Runtime machinery shared by built-in implicit schemes.

    Implicit schemes share dynamics binding, a translation workspace, a block
    allocator for one-stage resolvent requests, and a tableau/resolvent
    compatibility check. The scheme owns this object and may copy selected
    attributes onto itself to keep stage algorithms direct.
    """

    __slots__ = ("block_allocator", "dynamics", "workspace")

    def __init__(
        self,
        scheme: SchemeRuntimeImplicitOwner,
        dynamics: DynamicsLike,
        allocator: Allocator,
    ) -> None:
        self.validate_resolvent_tableau(scheme)
        translation_probe = allocator.allocate_translation()
        self.dynamics = dynamics
        self.workspace = SchemeStepSupport(allocator, translation_probe)
        self.block_allocator = BlockAllocator(allocator)

    @staticmethod
    def validate_resolvent_tableau(scheme: SchemeRuntimeImplicitOwner) -> None:
        """Check any declared resolvent tableau against the scheme tableau."""

        resolvent = getattr(scheme, "resolvent", None)
        scheme_tableau = getattr(scheme, "tableau", None)
        resolvent_tableau = getattr(resolvent, "tableau", None)

        if (
            scheme_tableau is not None
            and resolvent_tableau is not None
            and resolvent_tableau != scheme_tableau
        ):
            scheme_name = getattr(
                getattr(scheme, "descriptor", None),
                "short_name",
                type(scheme).__name__,
            )
            raise ValueError(
                f"{scheme_name} requires a resolvent configured with its own tableau."
            )

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)


__all__ = [
    "SchemeRuntimeImplicit",
]
