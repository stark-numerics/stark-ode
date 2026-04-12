from __future__ import annotations

from collections.abc import Callable

from stark.contracts import Combine2, Combine3, Combine4, Combine5, Combine6, Combine7, Scale, Translation

LinearCombine = tuple[Scale | Combine2 | Combine3 | Combine4 | Combine5 | Combine6 | Combine7, ...]


def fallback_scale(y: Translation, a: float, x: Translation) -> Translation:
    return a * x


def fallback_combine2(
    y: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
) -> Translation:
    return a0 * x0 + a1 * x1


def fallback_combine3(
    y: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
) -> Translation:
    return a0 * x0 + a1 * x1 + a2 * x2


def fallback_combine4(
    y: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    a3: float,
    x3: Translation,
) -> Translation:
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3


def fallback_combine5(
    y: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    a3: float,
    x3: Translation,
    a4: float,
    x4: Translation,
) -> Translation:
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4


def fallback_combine6(
    y: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    a3: float,
    x3: Translation,
    a4: float,
    x4: Translation,
    a5: float,
    x5: Translation,
) -> Translation:
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5


def fallback_combine7(
    y: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    a3: float,
    x3: Translation,
    a4: float,
    x4: Translation,
    a5: float,
    x5: Translation,
    a6: float,
    x6: Translation,
) -> Translation:
    return (
        a0 * x0
        + a1 * x1
        + a2 * x2
        + a3 * x3
        + a4 * x4
        + a5 * x5
        + a6 * x6
    )


def complete_linear_combine(
    linear_combine: LinearCombine,
    allocate_translation: Callable[[], Translation],
) -> tuple[Scale, Combine2, Combine3, Combine4, Combine5, Combine6, Combine7]:
    """
    Fill in missing high-arity combine kernels from lower-arity fast paths.

    The pure fallback functions below are deliberately simple and allocation
    heavy because the generic `Translation` contract only promises `__add__`
    and `__rmul__`. Once a user supplies `combine2`, however, STARK can build
    higher arities from it using scratch translations allocated once per
    scheme.
    """
    scale = linear_combine[0] if len(linear_combine) >= 1 else fallback_scale
    combine2 = linear_combine[1] if len(linear_combine) >= 2 else fallback_combine2

    if len(linear_combine) < 2:
        return (
            scale,
            combine2,
            fallback_combine3,
            fallback_combine4,
            fallback_combine5,
            fallback_combine6,
            fallback_combine7,
        )

    combine3 = (
        linear_combine[2]
        if len(linear_combine) >= 3
        else _compose_combine3(combine2, allocate_translation)
    )
    combine4 = (
        linear_combine[3]
        if len(linear_combine) >= 4
        else _compose_combine4(combine2, allocate_translation)
    )
    combine5 = (
        linear_combine[4]
        if len(linear_combine) >= 5
        else _compose_combine5(combine2, combine3, allocate_translation)
    )
    combine6 = (
        linear_combine[5]
        if len(linear_combine) >= 6
        else _compose_combine6(combine2, combine3, allocate_translation)
    )
    combine7 = (
        linear_combine[6]
        if len(linear_combine) >= 7
        else _compose_combine7(combine2, combine3, combine4, allocate_translation)
    )

    return scale, combine2, combine3, combine4, combine5, combine6, combine7


def _compose_combine3(
    combine2: Combine2,
    allocate_translation: Callable[[], Translation],
) -> Combine3:
    left = allocate_translation()

    def combine3(
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
    ) -> Translation:
        left_value = combine2(left, a0, x0, a1, x1)
        return combine2(out, 1.0, left_value, a2, x2)

    return combine3


def _compose_combine4(
    combine2: Combine2,
    allocate_translation: Callable[[], Translation],
) -> Combine4:
    left = allocate_translation()
    right = allocate_translation()

    def combine4(
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
    ) -> Translation:
        left_value = combine2(left, a0, x0, a1, x1)
        right_value = combine2(right, a2, x2, a3, x3)
        return combine2(out, 1.0, left_value, 1.0, right_value)

    return combine4


def _compose_combine5(
    combine2: Combine2,
    combine3: Combine3,
    allocate_translation: Callable[[], Translation],
) -> Combine5:
    left = allocate_translation()
    right = allocate_translation()

    def combine5(
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
    ) -> Translation:
        left_value = combine2(left, a0, x0, a1, x1)
        right_value = combine3(right, a2, x2, a3, x3, a4, x4)
        return combine2(out, 1.0, left_value, 1.0, right_value)

    return combine5


def _compose_combine6(
    combine2: Combine2,
    combine3: Combine3,
    allocate_translation: Callable[[], Translation],
) -> Combine6:
    left = allocate_translation()
    right = allocate_translation()

    def combine6(
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        a5: float,
        x5: Translation,
    ) -> Translation:
        left_value = combine3(left, a0, x0, a1, x1, a2, x2)
        right_value = combine3(right, a3, x3, a4, x4, a5, x5)
        return combine2(out, 1.0, left_value, 1.0, right_value)

    return combine6


def _compose_combine7(
    combine2: Combine2,
    combine3: Combine3,
    combine4: Combine4,
    allocate_translation: Callable[[], Translation],
) -> Combine7:
    left = allocate_translation()
    right = allocate_translation()

    def combine7(
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        a5: float,
        x5: Translation,
        a6: float,
        x6: Translation,
    ) -> Translation:
        left_value = combine3(left, a0, x0, a1, x1, a2, x2)
        right_value = combine4(right, a3, x3, a4, x4, a5, x5, a6, x6)
        return combine2(out, 1.0, left_value, 1.0, right_value)

    return combine7


def resolve_linear_combine(
    translation: Translation,
) -> LinearCombine:
    linear_combine = getattr(translation, "linear_combine", None)
    if linear_combine is None:
        return ()

    if not isinstance(linear_combine, (list, tuple)):
        raise TypeError("Translation.linear_combine must be a list or tuple of callables.")

    for index, combine in enumerate(linear_combine):
        if not callable(combine):
            arity_name = "scale" if index == 0 else f"combine{index + 1}"
            raise TypeError(f"Translation.linear_combine[{index}] must be a callable {arity_name}.")

    return tuple(linear_combine)


__all__ = [
    "LinearCombine",
    "complete_linear_combine",
    "fallback_combine2",
    "fallback_combine3",
    "fallback_combine4",
    "fallback_combine5",
    "fallback_combine6",
    "fallback_combine7",
    "fallback_scale",
    "resolve_linear_combine",
]
