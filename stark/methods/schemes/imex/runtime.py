from __future__ import annotations

from stark.core.auditor import Auditor
from stark.core.contracts import DynamicsSplitLike, State, AllocatorLike
from stark.methods.schemes.execution.step_support import SchemeStepSupport


class SchemeRuntimeImex:
    """Runtime machinery shared by built-in IMEX schemes.

    IMEX schemes use split dynamics, so setup audits both dynamics parts
    against the allocator before constructing the shared step workspace. The
    scheme owns this object and may copy the workspace onto itself for direct
    access in the stage algorithm.
    """

    __slots__ = ("workspace",)

    def __init__(
        self,
        dynamics: DynamicsSplitLike,
        allocator: AllocatorLike,
    ) -> None:
        translation_probe = allocator.allocate_translation()
        Auditor.require_imex_scheme_inputs(dynamics, allocator, translation_probe)
        self.workspace = SchemeStepSupport(allocator, translation_probe)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)


__all__ = [
    "SchemeRuntimeImex",
]
