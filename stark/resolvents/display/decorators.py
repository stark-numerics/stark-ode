from __future__ import annotations


def with_resolvent_display(cls):
    """Install standard resolvent display and metadata methods."""

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    def __repr__(self) -> str:
        extra_parts = []

        if hasattr(self, "depth"):
            extra_parts.append(f"depth={self.depth!r}")

        if hasattr(self, "inverter"):
            extra_parts.append(f"inverter={self.inverter!r}")

        extra = ", ".join(extra_parts)
        if extra:
            extra = f", {extra}"

        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance!r}, "
            f"maximum_steps={self.max_iterations!r}"
            f"{extra}, "
            f"accelerator={self.accelerator!r}, "
            f"tableau={self.tableau!r})"
        )

    def __str__(self) -> str:
        return type(self).__name__

    cls.short_name = short_name
    cls.__repr__ = __repr__
    cls.__str__ = __str__
    return cls


__all__ = ["with_resolvent_display"]
