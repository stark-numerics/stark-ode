from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(slots=True)
class AlgebraistSource:
    """Owned source record for generated Algebraist functions."""

    kernels: dict[str, str] = field(default_factory=dict)
    wrappers: dict[str, str] = field(default_factory=dict)

    def record_kernel(self, name: str, source: str) -> None:
        self.kernels[name] = source

    def record_wrapper(self, name: str, source: str) -> None:
        self.wrappers[name] = source

    @property
    def kernel_sources(self) -> MappingProxyType[str, str]:
        return MappingProxyType(self.kernels)

    @property
    def wrapper_sources(self) -> MappingProxyType[str, str]:
        return MappingProxyType(self.wrappers)

    @property
    def sources(self) -> MappingProxyType[str, str]:
        return MappingProxyType({**self.kernels, **self.wrappers})


__all__ = ["AlgebraistSource"]