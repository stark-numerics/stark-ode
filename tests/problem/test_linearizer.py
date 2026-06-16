from __future__ import annotations

from types import SimpleNamespace

from stark.problem.linearizer import Linearizer, LinearizerStyle
from stark.problem.system.system import System


class RecordingAccelerator:
    name = "recording"
    strict = False

    def __init__(self) -> None:
        self.labels: list[str | None] = []

    def compile(self, function=None, /, *, label=None, cache=None, **options):
        del cache, options

        def compile_function(target):
            self.labels.append(label)
            return target

        if function is None:
            return compile_function
        return compile_function(function)

    def compile_examples(self, function, *examples):
        del examples
        return function


class Translation:
    def __init__(self, dy):
        self.dy = list(dy)


class Operator:
    apply = None
    dense_fill = None


def test_linearizer_operator_configures_apply_and_dense_fill() -> None:
    def apply_kernel(y, source_dy, out_dy):
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
    operator = Operator()
    state = SimpleNamespace(y=[2.0, 3.0])

    linearizer(SimpleNamespace(present=0.0), state, operator)

    result = Translation([0.0, 0.0])
    operator.apply(Translation([5.0, 7.0]), result)
    assert result.dy == [10.0, 21.0]

    matrix = [0.0, 0.0, 0.0, 0.0]
    operator.dense_fill(None, matrix, 0, 0, 2)
    assert matrix == [2.0, 0.0, 0.0, 3.0]


def test_linearizer_kernel_returning_assigns_target_fields() -> None:
    @LinearizerStyle.kernel_returning(state=("y",), source=("dy",), target=("dy",))
    def apply_kernel(y, source_dy):
        return [y[0] * source_dy[0], y[1] * source_dy[1]]

    linearizer = Linearizer(apply_kernel)
    operator = Operator()
    state = SimpleNamespace(y=[2.0, 3.0])
    linearizer(SimpleNamespace(present=0.0), state, operator)

    result = Translation([0.0, 0.0])
    operator.apply(Translation([5.0, 7.0]), result)
    assert result.dy == [10.0, 21.0]


def test_system_prepare_linearizer_uses_accelerator() -> None:
    def apply_kernel(y, source_dy, out_dy):
        out_dy[0] = y[0] * source_dy[0]

    signature = LinearizerStyle.kernel(
        apply_kernel,
        state=("y",),
        source=("dy",),
        target=("dy",),
    )
    system = System(derivative=lambda t, state: state, frame=object(), linearizer=signature)
    accelerator = RecordingAccelerator()

    prepared = system.prepare_linearizer(SimpleNamespace(accelerator=accelerator))

    assert prepared is not None
    assert accelerator.labels == [None]
