from dataclasses import dataclass, field
from typing import Protocol


class IntegratorConfiguration(Protocol):
    check_progress: bool


@dataclass(frozen=True, slots=True)
class IntegratorConfigurationDefault:
    check_progress: bool = False
