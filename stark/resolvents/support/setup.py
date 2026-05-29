from __future__ import annotations

from stark.accelerators import AcceleratorAbsent
from stark.contracts import AcceleratorLike
from stark.resolvents.support.safety import ResolventSafety, ResolventSafetyDefault


def initialise_resolvent_runtime(
    resolvent,
    safety: ResolventSafety | None = None,
    accelerator: AcceleratorLike | None = None,
) -> None:
    resolvent.safety = safety if safety is not None else ResolventSafetyDefault()
    resolvent.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
    resolvent.alpha = 0.0
    resolvent._monitor = None

    if hasattr(resolvent, "call_inline"):
        resolvent.call_pure = resolvent.call_inline
        resolvent.redirect_call = resolvent.call_pure
    else:
        resolvent.interval = None
        resolvent.state = None
        resolvent.redirect_call = resolvent.call_unbound


def refresh_resolvent_call(resolvent) -> None:
    resolvent.redirect_call = resolvent.call_pure


__all__ = ["initialise_resolvent_runtime", "refresh_resolvent_call"]
