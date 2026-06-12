from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

from stark import Configuration, Tolerance
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.methods.inverters.relaxation import InverterRelaxationJacobi
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


BLOCK_SIZE = 32
REPEAT = 7
SOLVES_PER_SAMPLE = 500


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


@dataclass(slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source: ScalarTranslation, target: ScalarTranslation) -> None:
        target.value = self.scale * source.value

    def inverse(self, source: ScalarTranslation, target: ScalarTranslation) -> None:
        target.value = source.value / self.scale


def invert_entry(
    operator: ScaleEntryOperator,
    source: ScalarTranslation,
    target: ScalarTranslation,
) -> None:
    operator.inverse(source, target)


def zero_output(output: Block[ScalarTranslation]) -> None:
    for item in output:
        item.value = 0.0


def build_request() -> ResolventInverterRequest[ScalarTranslation]:
    operators = [ScaleEntryOperator(2.0 + (index % 5)) for index in range(BLOCK_SIZE)]
    residual = Block([ScalarTranslation(float(index + 1)) for index in range(BLOCK_SIZE)])
    return ResolventInverterRequest(operator=BlockOperatorDiagonal(operators), residual=residual)


def time_sample() -> float:
    request = build_request()
    output = Block([ScalarTranslation(0.0) for _ in range(BLOCK_SIZE)])
    inverter = InverterRelaxationJacobi[ScalarTranslation](
        invert_entry,
        configuration=Configuration(
            inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
            inverter_maximum_steps=2,
        ),
    )

    start = perf_counter()
    for _ in range(SOLVES_PER_SAMPLE):
        zero_output(output)
        inverter(request, output)
    stop = perf_counter()
    return (stop - start) / SOLVES_PER_SAMPLE


def main() -> None:
    samples = [time_sample() for _ in range(REPEAT)]
    microseconds = [sample * 1.0e6 for sample in samples]
    print("InverterRelaxationJacobi benchmark")
    print(f"block size:          {BLOCK_SIZE}")
    print(f"solves per sample:   {SOLVES_PER_SAMPLE}")
    print(f"best us per solve:   {min(microseconds):.3f}")
    print(f"median us per solve: {median(microseconds):.3f}")
    print(f"worst us per solve:  {max(microseconds):.3f}")


if __name__ == "__main__":
    main()
