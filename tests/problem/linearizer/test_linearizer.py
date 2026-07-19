from __future__ import annotations

from collections.abc import Callable, MutableSequence
from types import SimpleNamespace
from typing import Any, ClassVar, overload

from stark.engines import EngineNumpy
from stark import Frame
from stark.core.contracts.engines.accelerator import AcceleratorTarget
from stark.problem.linearizer import Linearizer, LinearizerStyle
from stark.problem.system.system import System


class DummyRecordingAccelerator:
    name: ClassVar[str] = "recording"
    strict = False

    def __init__(self) -> None:
        self.labels: list[str | None] = []

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

        def compile_function(target: AcceleratorTarget) -> AcceleratorTarget:
            self.labels.append(label)
            return target

        if function is None:
            return compile_function
        return compile_function(function)

    def compile_examples(
        self,
        function: AcceleratorTarget,
        *examples: Any,
    ) -> AcceleratorTarget:
        del examples
        return function


class DummyLinearizerTranslation:
    def __init__(self, dy):
        self.dy = list(dy)


class DummyLinearizerOperator:
    """TranslationOperator shell whose callables are installed by the linearizer."""

    def __init__(self) -> None:
        self.apply: Callable[
            [DummyLinearizerTranslation, DummyLinearizerTranslation],
            None,
        ] = self.unconfigured_apply
        self.dense_fill: Callable[
            [object, MutableSequence[float], int, int, int],
            None,
        ] = self.unconfigured_dense_fill

    @staticmethod
    def unconfigured_apply(
        source: DummyLinearizerTranslation,
        target: DummyLinearizerTranslation,
    ) -> None:
        del source, target
        raise AssertionError("linearizer did not configure apply")

    @staticmethod
    def unconfigured_dense_fill(
        basis: object,
        matrix: MutableSequence[float],
        row_offset: int,
        column_offset: int,
        stride: int,
    ) -> None:
        del basis, matrix, row_offset, column_offset, stride
        raise AssertionError("linearizer did not configure dense_fill")


def test_linearizer_operator_configures_apply_and_dense_fill() -> None:
    def apply_kernel(t, y, source_dy, out_dy):
        del t
        out_dy[0] = y[0] * source_dy[0]
        out_dy[1] = y[1] * source_dy[1]

    def dense_kernel(y, matrix, row_offset, column_offset, stride):
        matrix[(row_offset + 0) * stride + column_offset + 0] = y[0]
        matrix[(row_offset + 1) * stride + column_offset + 1] = y[1]

    signature = LinearizerStyle.operator(
        apply=apply_kernel,
        dense=dense_kernel,
        state=("y",),
        source=("dy",),
        target=("dy",),
    )
    linearizer = Linearizer(signature)
    operator = DummyLinearizerOperator()
    state = SimpleNamespace(y=[2.0, 3.0])

    linearizer(SimpleNamespace(present=0.0), state, operator)

    result = DummyLinearizerTranslation([0.0, 0.0])
    operator.apply(DummyLinearizerTranslation([5.0, 7.0]), result)
    assert result.dy == [10.0, 21.0]

    matrix = [0.0, 0.0, 0.0, 0.0]
    operator.dense_fill(None, matrix, 0, 0, 2)
    assert matrix == [2.0, 0.0, 0.0, 3.0]


def test_linearizer_kernel_returning_assigns_target_fields() -> None:
    @LinearizerStyle.kernel_accepts_instant_returns(state=("y",), source=("dy",), target=("dy",))
    def apply_kernel(t, y, source_dy):
        del t
        return [y[0] * source_dy[0], y[1] * source_dy[1]]

    linearizer = Linearizer(apply_kernel)
    operator = DummyLinearizerOperator()
    state = SimpleNamespace(y=[2.0, 3.0])
    linearizer(SimpleNamespace(present=0.0), state, operator)

    result = DummyLinearizerTranslation([0.0, 0.0])
    operator.apply(DummyLinearizerTranslation([5.0, 7.0]), result)
    assert result.dy == [10.0, 21.0]


def test_system_prepare_linearizer_uses_accelerator() -> None:
    def apply_kernel(t, y, source_dy, out_dy):
        del t
        out_dy[0] = y[0] * source_dy[0]

    signature = LinearizerStyle.kernel_accepts_instant_writes(
        apply_kernel,
        state=("y",),
        source=("dy",),
        target=("dy",),
    )
    frame = Frame.vector("y", translation="dy", length=1)
    system = System(dynamics=lambda t, state: state, frame=frame, linearizer=signature)
    accelerator = DummyRecordingAccelerator()
    engine = EngineNumpy(frame, accelerator=accelerator)

    prepared = system.prepare_linearizer(engine)

    assert prepared is not None
    assert "linearizer-apply" in accelerator.labels
