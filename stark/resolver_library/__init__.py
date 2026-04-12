"""Built-in nonlinear residual resolvers."""

from stark.resolver_library.newton import ResolverNewton
from stark.resolver_library.picard import ResolverPicard

__all__ = ["ResolverNewton", "ResolverPicard"]
