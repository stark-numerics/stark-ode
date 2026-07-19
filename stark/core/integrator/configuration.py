from dataclasses import dataclass
from typing import Protocol


class IntegratorConfigurationLike(Protocol):
    """Configuration shape shared by integrator implementations."""

    @property
    def check_progress(self) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class IntegratorConfigurationDefault:
    check_progress: bool = False
