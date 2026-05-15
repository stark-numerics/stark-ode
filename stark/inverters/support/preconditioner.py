from __future__ import annotations

from stark.contracts import Block, PreconditionerLike
from stark.block.operator import BlockOperator
from stark.inverters.support.workspace import InverterWorkspace


class Preconditioner:
    """
    Optional preconditioning worker with an identity fallback.

    The user-facing preconditioner can be any configured worker satisfying
    `PreconditionerLike`. This support object gives the inverter layer one
    house-style place to handle:

    - the no-preconditioner identity path
    - operator binding
    - optional size preparation
    """

    __slots__ = ("worker", "workspace", "_apply")

    def __init__(self, workspace: InverterWorkspace, worker: PreconditionerLike | None = None) -> None:
        self.workspace = workspace
        self.worker = worker
        self._apply = self._copy if worker is None else worker

    def __repr__(self) -> str:
        return f"Preconditioner(worker={self.worker!r})"

    def __str__(self) -> str:
        return "identity preconditioner" if self.worker is None else f"preconditioner {type(self.worker).__name__}"

    def bind(self, operator: BlockOperator) -> None:
        if self.worker is not None:
            self.worker.bind(operator)

    def prepare(self, size: int) -> None:
        if self.worker is not None and hasattr(self.worker, "prepare"):
            self.worker.prepare(size)

    def __call__(self, rhs: Block, out: Block) -> None:
        self._apply(rhs, out)

    def _copy(self, rhs: Block, out: Block) -> None:
        self.workspace.copy_block(out, rhs)


__all__ = ["Preconditioner"]










