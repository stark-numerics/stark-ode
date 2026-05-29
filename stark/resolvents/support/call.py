from __future__ import annotations

from stark.block import Block
from stark.contracts import AcceleratorLike, IntervalLike, State


def with_resolvent_call_methods(cls):
    """Install resolvent call routing."""

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator

    def __call__(self, problem, delta):
        return self.redirect_call(problem, delta)

    cls.bind_accelerator = bind_accelerator
    cls.__call__ = __call__
    return cls


def with_resolvent_binding_methods(cls):
    """Install legacy bound/unbound call routing."""

    def bind(self, interval: IntervalLike, state: State) -> None:
        self.interval = interval
        self.state = state
        self.redirect_call = (
            self.call_checked if self.safety.block_sizes else self.call_unchecked
        )

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator

    def __call__(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.redirect_call(alpha, rhs, out)

    def call_unbound(self, alpha: float, rhs: Block | None, out: Block) -> None:
        del alpha, rhs, out
        raise RuntimeError(f"{type(self).__name__} must be bound before use.")

    cls.bind = bind
    cls.bind_accelerator = bind_accelerator
    cls.__call__ = __call__
    cls.call_unbound = call_unbound
    return cls


__all__ = ["with_resolvent_binding_methods", "with_resolvent_call_methods"]
