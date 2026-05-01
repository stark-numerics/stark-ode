from __future__ import annotations

from collections.abc import Callable

from stark.contracts.acceleration import AcceleratorLike


def build_function(
    name: str,
    source: str,
    *,
    accelerator: AcceleratorLike | None = None,
    namespace: dict[str, object] | None = None,
) -> Callable[..., object]:
    local_namespace: dict[str, object] = {} if namespace is None else dict(namespace)
    exec(source, local_namespace)
    function = local_namespace[name]
    if accelerator is None:
        return function
    return accelerator.decorate(cache=False)(function)
