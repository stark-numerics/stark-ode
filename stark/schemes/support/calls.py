from __future__ import annotations


def unbound_scheme_call(*_args, **_kwargs):
    raise RuntimeError("Generated scheme call has not been bound.")


__all__ = ["unbound_scheme_call"]
