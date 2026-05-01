from __future__ import annotations

from collections.abc import Sequence


def combine_signature(term_count: int, probes: Sequence[object]) -> tuple[object, ...]:
    arguments: list[object] = list(probes)
    for _ in range(term_count):
        arguments.append(1.0)
        arguments.extend(probes)
    return tuple(arguments)


def apply_signature(probes: Sequence[object]) -> tuple[object, ...]:
    return tuple(probes) + tuple(probes) + tuple(probes)
