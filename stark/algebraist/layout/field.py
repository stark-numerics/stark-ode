from __future__ import annotations

from dataclasses import dataclass, field

from stark.algebraist.layout.path import AlgebraistLayoutPath, AlgebraistLayoutPathLike
from stark.algebraist.layout.policy import AlgebraistLayoutBroadcast, AlgebraistLayoutPolicy


@dataclass(frozen=True, slots=True)
class AlgebraistLayoutField:
    """One logical field in an Algebraist layout."""

    translation_path: AlgebraistLayoutPath | AlgebraistLayoutPathLike
    state_path: AlgebraistLayoutPath | AlgebraistLayoutPathLike
    policy: AlgebraistLayoutPolicy = field(default_factory=AlgebraistLayoutBroadcast)
    include_in_norm: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.translation_path, AlgebraistLayoutPath):
            object.__setattr__(
                self,
                "translation_path",
                AlgebraistLayoutPath(self.translation_path),
            )
        if not isinstance(self.state_path, AlgebraistLayoutPath):
            object.__setattr__(self, "state_path", AlgebraistLayoutPath(self.state_path))

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
