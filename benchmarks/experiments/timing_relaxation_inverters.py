"""Time current inverter families on the same diagonal block request.

This experiment is intentionally small and local. It asks whether relaxation
inverters are worth publicising as user-facing examples, or whether they should
stay in the advanced/internal toolbox until a real problem demonstrates their
value.

Run from the ``stark-ode`` directory with:

    python -m benchmarks.experiments.timing_relaxation_inverters
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from statistics import median
from time import perf_counter

from stark import Configuration, Frame, Tolerance
from stark.core.block import Block, BlockBasis
from stark.core.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods import (
    InverterDense,
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
)
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(frozen=True, slots=True)
class TimingRow:
    name: str
    first: float
    median_repeat: float
    minimum_repeat: float
    error: float


@dataclass(frozen=True, slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source, target) -> None:
        target.dx[:] = self.scale * source.dx

    def inverse(self, source, target) -> None:
        target.dx[:] = source.dx / self.scale

    def dense_fill(self, _basis, matrix: list[float], row: int, column: int, stride: int) -> None:
        matrix[row * stride + column] = self.scale


def invert_entry(operator: ScaleEntryOperator, source, target) -> None:
    operator.inverse(source, target)


def scale_values(blocks: int) -> list[float]:
    cycle = (1.5, 2.0, 2.5, 3.0)
    return [cycle[index % len(cycle)] for index in range(blocks)]


def allocate_block(engine: EngineNumpy, values: list[float] | None = None) -> Block:
    block = Block([engine.allocator.allocate_translation() for _ in range(len(values or []))])
    if values is not None:
        for translation, value in zip(block, values, strict=True):
            translation.dx[0] = value
    return block


def zero_block(block: Block) -> None:
    for translation in block:
        translation.dx[:] = 0.0


def error_against_expected(output: Block, residuals: list[float], scales: list[float]) -> float:
    return max(
        abs(float(translation.dx[0]) - residual / scale)
        for translation, residual, scale in zip(output, residuals, scales, strict=True)
    )


def time_request_inverter(
    name: str,
    inverter,
    request: ResolventInverterRequest,
    output: Block,
    residuals: list[float],
    scales: list[float],
    repeats: int,
) -> TimingRow:
    zero_block(output)
    start = perf_counter()
    inverter(request, output)
    first = perf_counter() - start

    timings: list[float] = []
    for _ in range(repeats):
        zero_block(output)
        start = perf_counter()
        inverter(request, output)
        timings.append(perf_counter() - start)

    return TimingRow(
        name=name,
        first=first,
        median_repeat=median(timings),
        minimum_repeat=min(timings),
        error=error_against_expected(output, residuals, scales),
    )


def time_instance_inverter(
    name: str,
    instance,
    residual: Block,
    output: Block,
    residuals: list[float],
    scales: list[float],
    repeats: int,
) -> TimingRow:
    zero_block(output)
    start = perf_counter()
    instance(residual, output)
    first = perf_counter() - start

    timings: list[float] = []
    for _ in range(repeats):
        zero_block(output)
        start = perf_counter()
        instance(residual, output)
        timings.append(perf_counter() - start)

    return TimingRow(
        name=name,
        first=first,
        median_repeat=median(timings),
        minimum_repeat=min(timings),
        error=error_against_expected(output, residuals, scales),
    )


def print_table(rows: list[TimingRow]) -> None:
    print("Relaxation inverter timing experiment")
    print("-------------------------------------")
    print("solver              | first    | median   | min      | max error")
    print("--------------------+----------+----------+----------+----------")
    for row in rows:
        print(
            f"{row.name:<19} | "
            f"{row.first:0.6f}s | "
            f"{row.median_repeat:0.6f}s | "
            f"{row.minimum_repeat:0.6f}s | "
            f"{row.error:0.3e}"
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    engine = EngineNumpy(Frame.scalar("x", translation="dx"))
    scales = scale_values(args.blocks)
    residuals = [float(index + 1) for index in range(args.blocks)]
    operator = BlockOperatorDiagonal([ScaleEntryOperator(scale) for scale in scales])
    residual = allocate_block(engine, residuals)
    output = allocate_block(engine, [0.0 for _ in scales])
    request = ResolventInverterRequest(operator=operator, residual=residual)
    configuration = Configuration(
        inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
        inverter_maximum_steps=64,
    )
    basis = BlockBasis([engine.translation_basis() for _ in scales])

    dense = InverterDense(basis=basis)
    dense_instance = dense.instance(operator)
    jacobi = InverterRelaxationJacobi(
        invert_entry,
        configuration=configuration,
    )
    richardson = InverterRelaxationRichardson(
        damping=0.25,
        configuration=configuration,
    )
    rows = [
        time_request_inverter("Dense", dense, request, output, residuals, scales, args.repeats),
        time_instance_inverter("Dense instance", dense_instance, residual, output, residuals, scales, args.repeats),
        time_request_inverter("Jacobi", jacobi, request, output, residuals, scales, args.repeats),
        time_request_inverter("Richardson", richardson, request, output, residuals, scales, args.repeats),
    ]
    print_table(rows)


if __name__ == "__main__":
    main()
