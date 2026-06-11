from __future__ import annotations

from typing import Generic

from stark.contracts import (
    BlockLike,
    InverterOutputMode,
    InverterRequest,
    LegacyInverterLike,
    TranslationType,
)


class InverterLegacyAdapter(Generic[TranslationType]):
    """
    Adapt a bind-then-solve inverter to the request-shaped inverter protocol.

    Older STARK Krylov inverters are configured in two steps:

        inverter.bind(operator)
        inverter(rhs, output)

    Newer resolvents pass the operator and right-hand side together as an
    `InverterRequest`. This adapter keeps the legacy inverter usable while the
    inverter package is being moved over to the newer protocol.
    """

    __slots__ = ("legacy",)

    output_mode = InverterOutputMode.improve

    def __init__(self, legacy: LegacyInverterLike) -> None:
        self.legacy = legacy

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.legacy.bind(request.operator)
        self.legacy(request.residual, output)


__all__ = ["InverterLegacyAdapter"]
