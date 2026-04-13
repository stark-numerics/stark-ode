"""Built-in nonlinear residual resolvers."""

from stark.resolver_library.anderson import ResolverAnderson
from stark.resolver_library.broyden import ResolverBroyden
from stark.resolver_library.newton import ResolverNewton
from stark.resolver_library.picard import ResolverPicard

__all__ = ["ResolverAnderson", "ResolverBroyden", "ResolverNewton", "ResolverPicard"]
