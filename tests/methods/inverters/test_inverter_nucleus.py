from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, overload

import pytest

from stark.core.contracts.engines.accelerator import AcceleratorTarget
from stark.methods.inverters.nucleus import InverterNucleus


@dataclass(slots=True)
class RecordingAccelerator:
    """Minimal accelerator used to prove the nucleus asks for compilation."""

    strict: bool = False
    labels: list[str | None] = field(default_factory=list)
    call_count: int = 0

    name: ClassVar[str] = "numba"

    @overload
    def compile(
        self,
        function: None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[[AcceleratorTarget], AcceleratorTarget]:
        ...

    @overload
    def compile(
        self,
        function: AcceleratorTarget,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget:
        ...

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget | Callable[[AcceleratorTarget], AcceleratorTarget]:
        del cache, options
        self.labels.append(label)

        if function is None:
            return lambda target: target

        def wrapped(*args, **kwargs):
            self.call_count += 1
            return function(*args, **kwargs)

        return wrapped

    def compile_examples(
        self,
        function: AcceleratorTarget,
        *examples: Any,
    ) -> AcceleratorTarget:
        del examples
        return function


@dataclass(slots=True)
class UnsupportedAccelerator:
    strict: bool = False
    name: ClassVar[str] = "jax"

    @overload
    def compile(
        self,
        function: None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[[AcceleratorTarget], AcceleratorTarget]:
        ...

    @overload
    def compile(
        self,
        function: AcceleratorTarget,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget:
        ...

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget | Callable[[AcceleratorTarget], AcceleratorTarget]:
        del function, label, cache, options
        raise AssertionError("unsupported best-effort accelerators should not be compiled")

    def compile_examples(
        self,
        function: AcceleratorTarget,
        *examples: Any,
    ) -> AcceleratorTarget:
        del examples
        return function


def test_nucleus_solves_generic_four_by_four_system() -> None:
    nucleus = InverterNucleus(4)
    matrix = [
        4.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 2.0,
    ]
    expected = [1.0, 2.0, 3.0, 4.0]
    image = [
        4.0 * expected[0] + expected[1],
        expected[0] + 3.0 * expected[1] + expected[2],
        expected[1] + 2.0 * expected[2] + expected[3],
        expected[2] + 2.0 * expected[3],
    ]
    result = [0.0, 0.0, 0.0, 0.0]

    nucleus(matrix, image, result)

    assert result == pytest.approx(expected)


def test_nucleus_uses_numba_named_accelerator_for_generic_kernel() -> None:
    accelerator = RecordingAccelerator()
    nucleus = InverterNucleus(4, accelerator=accelerator)
    matrix = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    image = [1.0, 2.0, 3.0, 4.0]
    result = [0.0, 0.0, 0.0, 0.0]

    nucleus(matrix, image, result)

    assert accelerator.labels == ["inverter-nucleus-4x4"]
    assert accelerator.call_count == 1
    assert result == pytest.approx(image)


def test_nucleus_ignores_unsupported_best_effort_accelerator() -> None:
    nucleus = InverterNucleus(4, accelerator=UnsupportedAccelerator())

    assert nucleus.kernel is None


def test_nucleus_rejects_unsupported_strict_accelerator() -> None:
    with pytest.raises(RuntimeError, match="InverterNucleus acceleration"):
        InverterNucleus(4, accelerator=UnsupportedAccelerator(strict=True))


def test_nucleus_factor_reuses_fixed_four_by_four_matrix() -> None:
    nucleus = InverterNucleus(4)
    matrix = [
        4.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 2.0,
    ]
    factor = nucleus.factor(matrix)

    first_expected = [1.0, 2.0, 3.0, 4.0]
    first_image = [
        4.0 * first_expected[0] + first_expected[1],
        first_expected[0] + 3.0 * first_expected[1] + first_expected[2],
        first_expected[1] + 2.0 * first_expected[2] + first_expected[3],
        first_expected[2] + 2.0 * first_expected[3],
    ]
    second_expected = [-2.0, 0.5, 1.25, -0.75]
    second_image = [
        4.0 * second_expected[0] + second_expected[1],
        second_expected[0] + 3.0 * second_expected[1] + second_expected[2],
        second_expected[1] + 2.0 * second_expected[2] + second_expected[3],
        second_expected[2] + 2.0 * second_expected[3],
    ]
    result = [0.0, 0.0, 0.0, 0.0]

    factor(first_image, result)
    assert result == pytest.approx(first_expected)

    factor(second_image, result)
    assert result == pytest.approx(second_expected)


def test_nucleus_factor_caches_three_by_three_inverse() -> None:
    nucleus = InverterNucleus(3)
    matrix = [
        3.0, 1.0, 0.0,
        1.0, 4.0, 2.0,
        0.0, 2.0, 5.0,
    ]
    factor = nucleus.factor(matrix)
    expected = [2.0, -1.0, 0.5]
    image = [
        3.0 * expected[0] + expected[1],
        expected[0] + 4.0 * expected[1] + 2.0 * expected[2],
        2.0 * expected[1] + 5.0 * expected[2],
    ]
    result = [0.0, 0.0, 0.0]

    factor(image, result)

    assert result == pytest.approx(expected)
