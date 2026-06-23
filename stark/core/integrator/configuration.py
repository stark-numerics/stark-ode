from dataclasses import dataclass, field
from typing import Protocol


class IntegratorConfiguration(Protocol):
    """Configuration shape shared by integrator implementations."""

    @property
    def check_progress(self) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class IntegratorConfigurationDefault:
    check_progress: bool = False
