from __future__ import annotations

from dataclasses import dataclass, field

from stark.engines.shared.algebraist.frame.path import AlgebraistFieldPath, AlgebraistFieldPathLike
from stark.engines.shared.algebraist.frame.policy import AlgebraistFrameBroadcast, AlgebraistFramePolicy


@dataclass(frozen=True, slots=True, init=False)
class AlgebraistField:
    """One logical field in an Algebraist frame."""

    translation_path: AlgebraistFieldPath
    state_path: AlgebraistFieldPath
    shape: tuple[int, ...] | None
    policy: AlgebraistFramePolicy = field(default_factory=AlgebraistFrameBroadcast)

    def __init__(
        self,
        translation_path: AlgebraistFieldPath | AlgebraistFieldPathLike,
        state_path: AlgebraistFieldPath | AlgebraistFieldPathLike,
        policy: AlgebraistFramePolicy | None = None,
    ) -> None:
        resolved_policy = AlgebraistFrameBroadcast() if policy is None else policy
        object.__setattr__(
            self,
            "translation_path",
            translation_path
            if isinstance(translation_path, AlgebraistFieldPath)
            else AlgebraistFieldPath(translation_path),
        )
        object.__setattr__(
            self,
            "state_path",
            state_path
            if isinstance(state_path, AlgebraistFieldPath)
            else AlgebraistFieldPath(state_path),
        )
        object.__setattr__(
            self,
            "policy",
            resolved_policy,
        )
        object.__setattr__(
            self,
            "shape",
            getattr(resolved_policy, "shape", None),
        )

    @property
    def translation(self) -> AlgebraistFieldPath:
        return self.translation_path

    @property
    def state(self) -> AlgebraistFieldPath:
        return self.state_path

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
