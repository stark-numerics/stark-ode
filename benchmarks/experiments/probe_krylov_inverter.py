"""Probe the current Arnoldi Krylov inverter on simple diagonal systems.

This experiment is a correctness-first companion to the timing experiments. It
uses a diagonal block operator because the exact solution is obvious and because
the same request can be solved by the dense inverter as a baseline. The sweep is
intended to show whether Arnoldi fails generally, only for small restart
windows, or only when it is used without a helpful preconditioner.

Run from the ``stark-ode`` directory with:

    python -m benchmarks.experiments.probe_krylov_inverter
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from time import perf_counter

from stark import Frame, Tolerance
from stark.core.block import Block, BlockBasis
from stark.core.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods import InverterDense, InverterKrylovArnoldi, PreconditionerDiagonalInverse
from stark.methods.inverters.configuration import InverterConfigurationDefault
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(frozen=True, slots=True)
class ProbeRow:
    solver: str
    blocks: int
    restart: str
    preconditioner: str
    elapsed: float
    error: float

    @property
    def status(self) -> str:
        return "ok" if self.error <= 1.0e-8 else "bad"


@dataclass(frozen=True, slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source, target) -> None:
        target.dx[:] = self.scale * source.dx

    def inverse(self, source, target) -> None:
        target.dx[:] = source.dx / self.scale

    def dense_fill(self, _basis, matrix: list[float], row: int, column: int, stride: int) -> None:
        matrix[row * stride + column] = self.scale


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def scale_values(blocks: int) -> list[float]:
    cycle = (1.25, 1.5, 2.0, 2.5, 3.0)
    return [cycle[index % len(cycle)] for index in range(blocks)]


def allocate_block(engine: EngineNumpy, values: list[float]) -> Block:
    block = Block([engine.allocator.allocate_translation() for _ in values])
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


def time_dense(
    engine: EngineNumpy,
    blocks: int,
    request: ResolventInverterRequest,
    output: Block,
    residuals: list[float],
    scales: list[float],
) -> ProbeRow:
    basis = BlockBasis([engine.translation_basis() for _ in scales])
    inverter = InverterDense(basis=basis)
    zero_block(output)
    start = perf_counter()
    inverter(request, output)
    elapsed = perf_counter() - start
    return ProbeRow(
        solver="Dense",
        blocks=blocks,
        restart="-",
        preconditioner="-",
        elapsed=elapsed,
        error=error_against_expected(output, residuals, scales),
    )


def time_krylov(
    engine: EngineNumpy,
    blocks: int,
    restart: int,
    preconditioner: PreconditionerDiagonalInverse | None,
    request: ResolventInverterRequest,
    output: Block,
    residuals: list[float],
    scales: list[float],
    maximum_steps: int,
) -> ProbeRow:
    configuration = InverterConfigurationDefault(
        inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
        inverter_maximum_steps=maximum_steps,
    )
    inverter = InverterKrylovArnoldi(
        allocator=engine.allocator,
        inner_product=engine.allocator.inner_product,
        restart=restart,
        configuration=configuration,
        preconditioner=preconditioner,
    )
    zero_block(output)
    start = perf_counter()
    inverter(request, output)
    elapsed = perf_counter() - start
    return ProbeRow(
        solver="Arnoldi",
        blocks=blocks,
        restart=str(restart),
        preconditioner="diagonal" if preconditioner is not None else "none",
        elapsed=elapsed,
        error=error_against_expected(output, residuals, scales),
    )


def print_table(rows: list[ProbeRow]) -> None:
    print("Krylov inverter probe")
    print("---------------------")
    print("solver  | blocks | restart | precond  | elapsed  | max error | status")
    print("--------+--------+---------+----------+----------+-----------+-------")
    for row in rows:
        print(
            f"{row.solver:<7} | "
            f"{row.blocks:>6} | "
            f"{row.restart:>7} | "
            f"{row.preconditioner:<8} | "
            f"{row.elapsed:0.6f}s | "
            f"{row.error:0.3e} | "
            f"{row.status}"
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--blocks", default="1,2,4,8,16")
    parser.add_argument("--restarts", default="1,2,4,8,16")
    parser.add_argument("--maximum-steps", type=int, default=64)
    args = parser.parse_args()

    engine = EngineNumpy(Frame.scalar("x", translation="dx"))
    rows: list[ProbeRow] = []
    for blocks in parse_ints(args.blocks):
        scales = scale_values(blocks)
        residuals = [float(index + 1) for index in range(blocks)]
        operator = BlockOperatorDiagonal([ScaleEntryOperator(scale) for scale in scales])
        residual = allocate_block(engine, residuals)
        output = allocate_block(engine, [0.0 for _ in scales])
        request = ResolventInverterRequest(operator=operator, residual=residual)

        rows.append(time_dense(engine, blocks, request, output, residuals, scales))
        for restart in parse_ints(args.restarts):
            rows.append(
                time_krylov(
                    engine,
                    blocks,
                    restart,
                    None,
                    request,
                    output,
                    residuals,
                    scales,
                    args.maximum_steps,
                )
            )
            rows.append(
                time_krylov(
                    engine,
                    blocks,
                    restart,
                    PreconditionerDiagonalInverse(),
                    request,
                    output,
                    residuals,
                    scales,
                    args.maximum_steps,
                )
            )

    print_table(rows)


if __name__ == "__main__":
    main()
