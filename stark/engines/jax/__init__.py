"""JAX engine parts.

`EngineJax` stores frame fields in JAX arrays. The current integration focuses
on generated algebra kernels and backend correctness; whole-solver JIT is a
separate future target because adaptive control flow still lives in Python.
"""

from stark.engines.jax.allocator import EngineAllocatorJax
from stark.engines.jax.engine import EngineJax
from stark.engines.jax.translation import EngineTranslationJax

__all__ = [
    "EngineAllocatorJax",
    "EngineJax",
    "EngineTranslationJax",
]
