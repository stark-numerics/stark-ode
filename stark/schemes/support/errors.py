from __future__ import annotations

from stark.contracts.errors import StarkError

class SchemeErrorUnbound(StarkError):
    """
    Error raised by a scheme when a call is made to a generated method that has not been bound to a concrete function.
    """


def unbound_scheme_call(*_args, **_kwargs):
    raise SchemeErrorUnbound("Generated method has not been bound to a concrete function.")


__all__ = ["unbound_scheme_call"]
