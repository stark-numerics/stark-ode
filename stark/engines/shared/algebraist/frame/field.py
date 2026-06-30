from __future__ import annotations

from dataclasses import dataclass, field

from stark.engines.shared.algebraist.frame.norm import AlgebraistFrameNormPolicy, AlgebraistFrameNormRMS
from stark.engines.shared.algebraist.frame.path import AlgebraistFramePath, AlgebraistFramePathLike
from stark.engines.shared.algebraist.frame.policy import AlgebraistFrameBroadcast, AlgebraistFramePolicy


@dataclass(frozen=True, slots=True, init=False)
class AlgebraistFrameField:
    """One logical field in an Algebraist frame."""

    translation_path: AlgebraistFramePath
    state_path: AlgebraistFramePath
    policy: AlgebraistFramePolicy = field(default_factory=AlgebraistFrameBroadcast)
    norm: AlgebraistFrameNormPolicy = field(default_factory=AlgebraistFrameNormRMS)

    def __init__(
        self,
        translation_path: AlgebraistFramePath | AlgebraistFramePathLike,
        state_path: AlgebraistFramePath | AlgebraistFramePathLike,
        policy: AlgebraistFramePolicy | None = None,
        norm: AlgebraistFrameNormPolicy | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "translation_path",
            translation_path
            if isinstance(translation_path, AlgebraistFramePath)
            else AlgebraistFramePath(translation_path),
        )
        object.__setattr__(
            self,
            "state_path",
            state_path
            if isinstance(state_path, AlgebraistFramePath)
            else AlgebraistFramePath(state_path),
        )
        object.__setattr__(
            self,
            "policy",
            AlgebraistFrameBroadcast() if policy is None else policy,
        )
        object.__setattr__(
            self,
            "norm",
            AlgebraistFrameNormRMS() if norm is None else norm,
        )

    @property
    def translation_name(self) -> str:
        return self.translation_path.name

    @property
    def state_name(self) -> str:
        return self.state_path.name

    def translation_expression(self, root: str) -> str:
        return self.translation_path.expression(root)

    def state_expression(self, root: str) -> str:
        return self.state_path.expression(root)
