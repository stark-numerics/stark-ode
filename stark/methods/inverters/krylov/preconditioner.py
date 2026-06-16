from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.contracts import BlockLike, BlockOperatorLike, TranslationType


class InverterKrylovPreconditionerLike(Protocol[TranslationType]):
    """Apply an approximate inverse used by an `InverterKrylovArnoldi` solve.

    The preconditioner receives the current linear operator so implementations
    can update cached factors or inspect operator-specific structure without the
    Krylov inverter knowing those details.
    """

    def __call__(
        self,
        operator: BlockOperatorLike[TranslationType],
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        ...


__all__ = ["InverterKrylovPreconditionerLike"]
