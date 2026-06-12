from __future__ import annotations

from dataclasses import dataclass, field

from stark.engines.algebraist.frame.norm import AlgebraistFrameNormPolicy, AlgebraistFrameNormRMS
from stark.engines.algebraist.frame.path import AlgebraistFramePath, AlgebraistFramePathLike
from stark.engines.algebraist.frame.policy import AlgebraistFrameBroadcast, AlgebraistFramePolicy


@dataclass(frozen=True, slots=True)
class AlgebraistFrameField:
    """One logical field in an Algebraist frame."""

    translation_path: AlgebraistFramePath | AlgebraistFramePathLike
    state_path: AlgebraistFramePath | AlgebraistFramePathLike
    policy: AlgebraistFramePolicy = field(default_factory=AlgebraistFrameBroadcast)
    norm: AlgebraistFrameNormPolicy = field(default_factory=AlgebraistFrameNormRMS)

    def __post_init__(self) -> None:
        if not isinstance(self.translation_path, AlgebraistFramePath):
            object.__setattr__(
                self,
                "translation_path",
                AlgebraistFramePath(self.translation_path),
            )
        if not isinstance(self.state_path, AlgebraistFramePath):
            object.__setattr__(self, "state_path", AlgebraistFramePath(self.state_path))

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
