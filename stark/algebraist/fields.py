from __future__ import annotations

from dataclasses import dataclass, field

from stark.algebraist.paths import AlgebraistPath, normalize_path
from stark.algebraist.policies import AlgebraistBroadcast, AlgebraistPolicy


@dataclass(frozen=True, slots=True)
class AlgebraistField:
    """Describe one generated translation field."""

    translation_path: AlgebraistPath | str
    state_path: AlgebraistPath | str
    policy: AlgebraistPolicy = field(default_factory=AlgebraistBroadcast)
    include_in_norm: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation_path", normalize_path(self.translation_path))
        object.__setattr__(self, "state_path", normalize_path(self.state_path))
        object.__setattr__(self, "policy", self.policy.normalized())

    @property
    def translation_name(self) -> str:
        return "_".join(self.translation_path)

    @property
    def state_name(self) -> str:
        return "_".join(self.state_path)
