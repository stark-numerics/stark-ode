from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Safety:
    """Shared safety policy for STARK workers."""

    progress: bool = True
    block_sizes: bool = True
    apply_delta: bool = True

    def __repr__(self) -> str:
        return (
            "Safety("
            f"progress={self.progress!r}, "
            f"block_sizes={self.block_sizes!r}, "
            f"apply_delta={self.apply_delta!r})"
        )

    def __str__(self) -> str:
        return (
            f"progress={self.progress!r}, "
            f"block_sizes={self.block_sizes!r}, "
            f"apply_delta={self.apply_delta!r}"
        )

    @classmethod
    def fast(cls) -> "Safety":
        return cls(progress=False, block_sizes=False, apply_delta=False)


__all__ = ["Safety"]
