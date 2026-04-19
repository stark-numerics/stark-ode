from __future__ import annotations

from collections.abc import Callable

from stark.contracts import Combine2, Combine3, Combine4, Combine5, Combine6, Combine7, Scale, Translation

LinearCombine = tuple[Scale | Combine2 | Combine3 | Combine4 | Combine5 | Combine6 | Combine7, ...]


def fallback_scale(out: Translation, a: float, x: Translation) -> Translation:
    del out
    return a * x


def fallback_combine2(out: Translation, a0: float, x0: Translation, a1: float, x1: Translation) -> Translation:
    del out
    return a0 * x0 + a1 * x1


def fallback_combine3(
    out: Translation,
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
) -> Translation:
    del out
    return a0 * x0 + a1 * x1 + a2 * x2


def fallback_combine4(
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
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3


def fallback_combine5(
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
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4


def fallback_combine6(
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
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5


def fallback_combine7(
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
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6


class Combine3Worker:
    __slots__ = ("combine2", "left")

    def __init__(self, combine2: Combine2, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.left = allocate_translation()

    def __call__(self, out: Translation, a0: float, x0: Translation, a1: float, x1: Translation, a2: float, x2: Translation) -> Translation:
        left_value = self.combine2(self.left, a0, x0, a1, x1)
        return self.combine2(out, 1.0, left_value, a2, x2)


class Combine4Worker:
    __slots__ = ("combine2", "left", "right")

    def __init__(self, combine2: Combine2, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
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
        left_value = self.combine2(self.left, a0, x0, a1, x1)
        right_value = self.combine2(self.right, a2, x2, a3, x3)
        return self.combine2(out, 1.0, left_value, 1.0, right_value)


class Combine5Worker:
    __slots__ = ("combine2", "combine3", "left", "right")

    def __init__(self, combine2: Combine2, combine3: Combine3, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.combine3 = combine3
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
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
        left_value = self.combine2(self.left, a0, x0, a1, x1)
        right_value = self.combine3(self.right, a2, x2, a3, x3, a4, x4)
        return self.combine2(out, 1.0, left_value, 1.0, right_value)


class Combine6Worker:
    __slots__ = ("combine2", "combine3", "left", "right")

    def __init__(self, combine2: Combine2, combine3: Combine3, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.combine3 = combine3
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
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
        left_value = self.combine3(self.left, a0, x0, a1, x1, a2, x2)
        right_value = self.combine3(self.right, a3, x3, a4, x4, a5, x5)
        return self.combine2(out, 1.0, left_value, 1.0, right_value)


class Combine7Worker:
    __slots__ = ("combine2", "combine3", "combine4", "left", "right")

    def __init__(
        self,
        combine2: Combine2,
        combine3: Combine3,
        combine4: Combine4,
        allocate_translation: Callable[[], Translation],
    ) -> None:
        self.combine2 = combine2
        self.combine3 = combine3
        self.combine4 = combine4
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
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
        left_value = self.combine3(self.left, a0, x0, a1, x1, a2, x2)
        right_value = self.combine4(self.right, a3, x3, a4, x4, a5, x5, a6, x6)
        return self.combine2(out, 1.0, left_value, 1.0, right_value)


class Combiner:
    """
    Resolve a translation's available linear-combination kernels into one worker.

    The generic `Translation` contract only guarantees scalar multiplication and
    addition, so the pure fallbacks remain allocation-heavy. When a translation
    supplies even a small fast-path family, however, the combiner can compose
    higher-arity kernels once during setup and hand branch-free callables to the
    hot scheme, resolvent, and inverter paths.
    """

    __slots__ = ("scale", "combine2", "combine3", "combine4", "combine5", "combine6", "combine7")

    def __init__(self, linear_combine: LinearCombine, allocate_translation: Callable[[], Translation]) -> None:
        self.scale = linear_combine[0] if len(linear_combine) >= 1 else fallback_scale
        self.combine2 = linear_combine[1] if len(linear_combine) >= 2 else fallback_combine2

        if len(linear_combine) < 2:
            self.combine3 = fallback_combine3
            self.combine4 = fallback_combine4
            self.combine5 = fallback_combine5
            self.combine6 = fallback_combine6
            self.combine7 = fallback_combine7
            return

        self.combine3 = linear_combine[2] if len(linear_combine) >= 3 else Combine3Worker(self.combine2, allocate_translation)
        self.combine4 = linear_combine[3] if len(linear_combine) >= 4 else Combine4Worker(self.combine2, allocate_translation)
        self.combine5 = linear_combine[4] if len(linear_combine) >= 5 else Combine5Worker(self.combine2, self.combine3, allocate_translation)
        self.combine6 = linear_combine[5] if len(linear_combine) >= 6 else Combine6Worker(self.combine2, self.combine3, allocate_translation)
        self.combine7 = (
            linear_combine[6]
            if len(linear_combine) >= 7
            else Combine7Worker(self.combine2, self.combine3, self.combine4, allocate_translation)
        )

    def as_tuple(self) -> tuple[Scale, Combine2, Combine3, Combine4, Combine5, Combine6, Combine7]:
        return (self.scale, self.combine2, self.combine3, self.combine4, self.combine5, self.combine6, self.combine7)


def resolve_linear_combine(translation: Translation) -> LinearCombine:
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
    "Combiner",
    "LinearCombine",
    "fallback_combine2",
    "fallback_combine3",
    "fallback_combine4",
    "fallback_combine5",
    "fallback_combine6",
    "fallback_combine7",
    "fallback_scale",
    "resolve_linear_combine",
]
