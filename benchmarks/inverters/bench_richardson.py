from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.relaxation import InverterRelaxationRichardson
from stark.inverters.support import InverterBudget, InverterTolerance
from stark.resolvents.requests.inverter import ResolventInverterRequest


REPEAT = 7
SOLVES_PER_SAMPLE = 10_000
BLOCK_SIZE = 8


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin, result) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


def scale_by_two(source: ScalarTranslation, target: ScalarTranslation) -> None:
    target.value = 2.0 * source.value


def build_case() -> tuple[
    InverterRelaxationRichardson[ScalarTranslation],
    ResolventInverterRequest,
    Block[ScalarTranslation],
]:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=BLOCK_SIZE),
        residual=Block([ScalarTranslation(float(index + 1)) for index in range(BLOCK_SIZE)]),
    )
    output = Block([ScalarTranslation(0.0) for _ in range(BLOCK_SIZE)])
    inverter = InverterRelaxationRichardson[ScalarTranslation](
        damping=0.5,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=4),
    )
    return inverter, request, output


def zero_output(output: Block[ScalarTranslation]) -> None:
    for item in output:
        item.value = 0.0


def time_sample() -> float:
    inverter, request, output = build_case()
    start = perf_counter()
    for _ in range(SOLVES_PER_SAMPLE):
        zero_output(output)
        inverter(request, output)
    stop = perf_counter()
    return (stop - start) / SOLVES_PER_SAMPLE


def main() -> None:
    samples = [time_sample() for _ in range(REPEAT)]
    microseconds = [sample * 1.0e6 for sample in samples]
    print("InverterRelaxationRichardson benchmark")
    print(f"block size:          {BLOCK_SIZE}")
    print(f"solves per sample:   {SOLVES_PER_SAMPLE}")
    print(f"best us per solve:   {min(microseconds):.3f}")
    print(f"median us per solve: {median(microseconds):.3f}")
    print(f"worst us per solve:  {max(microseconds):.3f}")


if __name__ == "__main__":
    main()
