from __future__ import annotations

from stark.core.contracts.errors import StarkError

def unbound_scheme_call(*_args, **_kwargs):
    raise StarkError("Generated method has not been bound to a concrete function.")

__all__ = ["unbound_scheme_call"]
