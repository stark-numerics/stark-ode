from __future__ import annotations

class ResolventTableauGuard:
    """Check that a tableau-aware resolvent matches the calling scheme."""

    __slots__ = ("method_name", "tableau")

    def __init__(self, method_name: str, tableau) -> None:
        self.method_name = method_name
        self.tableau = tableau

    def __repr__(self) -> str:
        return f"ResolventTableauGuard(method_name={self.method_name!r}, tableau={self.tableau!r})"

    def __str__(self) -> str:
        return f"{self.method_name} resolvent tableau guard"

    def __call__(self, resolvent) -> None:
        candidate = getattr(resolvent, "tableau", None)
        if candidate is None or candidate == self.tableau:
            return

        candidate_name = getattr(candidate, "short_name", None) or type(candidate).__name__
        scheme_name = self.tableau.short_name if self.tableau.short_name is not None else self.method_name
        raise ValueError(
            f"{self.method_name} requires a compatible resolvent tableau; "
            f"expected {scheme_name!r}, got {candidate_name!r}."
        )


__all__ = ["ResolventTableauGuard"]









