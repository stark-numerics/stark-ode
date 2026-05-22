from __future__ import annotations

from collections.abc import Callable

from stark.contracts import (
    Combine2,
    Combine3,
    Combine4,
    Combine5,
    Combine6,
    Combine7,
    Combine8,
    Combine9,
    Combine10,
    Combine11,
    Combine12,
    LinearCombine,
    Scale,
    Translation,
)


def fallback_scale(a: float, x: Translation, out: Translation) -> Translation:
    del out
    return a * x


def fallback_combine2(a0: float, x0: Translation, a1: float, x1: Translation, out: Translation) -> Translation:
    del out
    return a0 * x0 + a1 * x1


def fallback_combine3(
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    out: Translation,
) -> Translation:
    del out
    return a0 * x0 + a1 * x1 + a2 * x2


def fallback_combine4(
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    a3: float,
    x3: Translation,
    out: Translation,
) -> Translation:
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3


def fallback_combine5(
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
    out: Translation,
) -> Translation:
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4


def fallback_combine6(
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
    out: Translation,
) -> Translation:
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5


def fallback_combine7(
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
    out: Translation,
) -> Translation:
    del out
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6


def _fallback_combine_many(*terms: object) -> Translation:
    out = terms[-1]
    terms = terms[:-1]
    del out
    if len(terms) % 2 != 0:
        raise TypeError("Linear combination terms must be coefficient/translation pairs.")
    if not terms:
        raise ValueError("Linear combination requires at least one term.")

    coefficient = terms[0]
    translation = terms[1]
    result = coefficient * translation
    for index in range(2, len(terms), 2):
        coefficient = terms[index]
        translation = terms[index + 1]
        result = result + coefficient * translation
    return result


def fallback_combine8(*terms: object) -> Translation:
    return _fallback_combine_many(*terms)


def fallback_combine9(*terms: object) -> Translation:
    return _fallback_combine_many(*terms)


def fallback_combine10(*terms: object) -> Translation:
    return _fallback_combine_many(*terms)


def fallback_combine11(*terms: object) -> Translation:
    return _fallback_combine_many(*terms)


def fallback_combine12(*terms: object) -> Translation:
    return _fallback_combine_many(*terms)


class AlgebraistCombineSynthesizer3:
    __slots__ = ("combine2", "left")

    def __init__(self, combine2: Combine2, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.left = allocate_translation()

    def __call__(self, a0: float, x0: Translation, a1: float, x1: Translation, a2: float, x2: Translation, out: Translation) -> Translation:
        left_value = self.combine2(a0, x0, a1, x1, self.left)
        return self.combine2(1.0, left_value, a2, x2, out)


class AlgebraistCombineSynthesizer4:
    __slots__ = ("combine2", "left", "right")

    def __init__(self, combine2: Combine2, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        out: Translation,
    ) -> Translation:
        left_value = self.combine2(a0, x0, a1, x1, self.left)
        right_value = self.combine2(a2, x2, a3, x3, self.right)
        return self.combine2(1.0, left_value, 1.0, right_value, out)


class AlgebraistCombineSynthesizer5:
    __slots__ = ("combine2", "combine3", "left", "right")

    def __init__(self, combine2: Combine2, combine3: Combine3, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.combine3 = combine3
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
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
        out: Translation,
    ) -> Translation:
        left_value = self.combine2(a0, x0, a1, x1, self.left)
        right_value = self.combine3(a2, x2, a3, x3, a4, x4, self.right)
        return self.combine2(1.0, left_value, 1.0, right_value, out)


class AlgebraistCombineSynthesizer6:
    __slots__ = ("combine2", "combine3", "left", "right")

    def __init__(self, combine2: Combine2, combine3: Combine3, allocate_translation: Callable[[], Translation]) -> None:
        self.combine2 = combine2
        self.combine3 = combine3
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(
        self,
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
        out: Translation,
    ) -> Translation:
        left_value = self.combine3(a0, x0, a1, x1, a2, x2, self.left)
        right_value = self.combine3(a3, x3, a4, x4, a5, x5, self.right)
        return self.combine2(1.0, left_value, 1.0, right_value, out)


class AlgebraistCombineSynthesizer7:
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
        out: Translation,
    ) -> Translation:
        left_value = self.combine3(a0, x0, a1, x1, a2, x2, self.left)
        right_value = self.combine4(a3, x3, a4, x4, a5, x5, a6, x6, self.right)
        return self.combine2(1.0, left_value, 1.0, right_value, out)


class AlgebraistCombineSynthesizerN:
    __slots__ = ("arity", "combine2", "left", "right")

    def __init__(self, arity: int, combine2: Combine2, allocate_translation: Callable[[], Translation]) -> None:
        if arity < 3:
            raise ValueError("AlgebraistCombineSynthesizerN requires arity >= 3.")
        self.arity = arity
        self.combine2 = combine2
        self.left = allocate_translation()
        self.right = allocate_translation()

    def __call__(self, *terms: object) -> Translation:
        if len(terms) != 2 * self.arity + 1:
            raise TypeError(
                f"combine{self.arity} requires {self.arity} coefficient/translation pairs."
            )

        out = terms[-1]
        terms = terms[:-1]
        total = self.combine2(terms[0], terms[1], terms[2], terms[3], self.left)
        target = self.right
        for index in range(4, len(terms), 2):
            is_last = index == len(terms) - 2
            target = out if is_last else (self.right if total is self.left else self.left)
            total = self.combine2(1.0, total, terms[index], terms[index + 1], target)
        return total


class AlgebraistCombineResolver:
    """
    Resolve a translation's available linear-combination kernels into one worker.

    The generic `Translation` contract only guarantees scalar multiplication and
    addition, so the pure fallbacks remain allocation-heavy. When a translation
    supplies even a small fast-path family, however, the AlgebraistCombineResolver can compose
    higher-arity kernels once during setup and hand branch-free callables to the
    hot scheme, resolvent, and inverter paths.
    """

    __slots__ = (
        "scale",
        "combine2",
        "combine3",
        "combine4",
        "combine5",
        "combine6",
        "combine7",
        "combine8",
        "combine9",
        "combine10",
        "combine11",
        "combine12",
    )

    def __init__(self, linear_combine: LinearCombine, allocate_translation: Callable[[], Translation]) -> None:
        self.scale = linear_combine[0] if len(linear_combine) >= 1 else fallback_scale
        self.combine2 = linear_combine[1] if len(linear_combine) >= 2 else fallback_combine2

        if len(linear_combine) < 2:
            self.combine3 = fallback_combine3
            self.combine4 = fallback_combine4
            self.combine5 = fallback_combine5
            self.combine6 = fallback_combine6
            self.combine7 = fallback_combine7
            self.combine8 = fallback_combine8
            self.combine9 = fallback_combine9
            self.combine10 = fallback_combine10
            self.combine11 = fallback_combine11
            self.combine12 = fallback_combine12
            return

        self.combine3 = linear_combine[2] if len(linear_combine) >= 3 else AlgebraistCombineSynthesizer3(self.combine2, allocate_translation)
        self.combine4 = linear_combine[3] if len(linear_combine) >= 4 else AlgebraistCombineSynthesizer4(self.combine2, allocate_translation)
        self.combine5 = linear_combine[4] if len(linear_combine) >= 5 else AlgebraistCombineSynthesizer5(self.combine2, self.combine3, allocate_translation)
        self.combine6 = linear_combine[5] if len(linear_combine) >= 6 else AlgebraistCombineSynthesizer6(self.combine2, self.combine3, allocate_translation)
        self.combine7 = (
            linear_combine[6]
            if len(linear_combine) >= 7
            else AlgebraistCombineSynthesizer7(self.combine2, self.combine3, self.combine4, allocate_translation)
        )
        self.combine8 = linear_combine[7] if len(linear_combine) >= 8 else AlgebraistCombineSynthesizerN(8, self.combine2, allocate_translation)
        self.combine9 = linear_combine[8] if len(linear_combine) >= 9 else AlgebraistCombineSynthesizerN(9, self.combine2, allocate_translation)
        self.combine10 = linear_combine[9] if len(linear_combine) >= 10 else AlgebraistCombineSynthesizerN(10, self.combine2, allocate_translation)
        self.combine11 = linear_combine[10] if len(linear_combine) >= 11 else AlgebraistCombineSynthesizerN(11, self.combine2, allocate_translation)
        self.combine12 = linear_combine[11] if len(linear_combine) >= 12 else AlgebraistCombineSynthesizerN(12, self.combine2, allocate_translation)

    def as_tuple(
        self,
    ) -> tuple[
        Scale,
        Combine2,
        Combine3,
        Combine4,
        Combine5,
        Combine6,
        Combine7,
        Combine8,
        Combine9,
        Combine10,
        Combine11,
        Combine12,
    ]:
        return (
            self.scale,
            self.combine2,
            self.combine3,
            self.combine4,
            self.combine5,
            self.combine6,
            self.combine7,
            self.combine8,
            self.combine9,
            self.combine10,
            self.combine11,
            self.combine12,
        )


    @classmethod
    def from_translation(cls, translation: Translation, allocate_translation: Callable[[], Translation]):
        linear_combine = getattr(translation, "linear_combine", None)

        if linear_combine is None:
            linear_combine = ()
        elif not isinstance(linear_combine, (list, tuple)):
            raise TypeError("Translation.linear_combine must be a list or tuple of callables.")
        else:
            for index, combine in enumerate(linear_combine):
                if not callable(combine):
                    arity_name = "scale" if index == 0 else f"combine{index + 1}"
                    raise TypeError(
                        f"Translation.linear_combine[{index}] must be a callable {arity_name}."
                    )
            linear_combine = tuple(linear_combine)

        return cls(linear_combine, allocate_translation)


__all__ = [
    "AlgebraistCombineResolver",
    "LinearCombine",
    "fallback_combine10",
    "fallback_combine11",
    "fallback_combine12",
    "fallback_combine2",
    "fallback_combine3",
    "fallback_combine4",
    "fallback_combine5",
    "fallback_combine6",
    "fallback_combine7",
    "fallback_combine8",
    "fallback_combine9",
    "fallback_scale",
]
