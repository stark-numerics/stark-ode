from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


REPEAT = 7
CALLS_PER_SAMPLE = 20_000
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


def build_case() -> tuple[InverterDefect[ScalarTranslation], ResolventInverterRequest, Block[ScalarTranslation]]:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=BLOCK_SIZE),
        residual=Block([ScalarTranslation(float(index + 1)) for index in range(BLOCK_SIZE)]),
    )
    output = Block([ScalarTranslation(0.25) for _ in range(BLOCK_SIZE)])
    defect = InverterDefect[ScalarTranslation]()
    defect(request, output)
    return defect, request, output


def time_sample() -> float:
    defect, request, output = build_case()
    start = perf_counter()
    for _ in range(CALLS_PER_SAMPLE):
        defect(request, output)
    stop = perf_counter()
    return (stop - start) / CALLS_PER_SAMPLE


def main() -> None:
    samples = [time_sample() for _ in range(REPEAT)]
    microseconds = [sample * 1.0e6 for sample in samples]
    print("InverterDefect benchmark")
    print(f"block size:          {BLOCK_SIZE}")
    print(f"calls per sample:    {CALLS_PER_SAMPLE}")
    print(f"best us per call:    {min(microseconds):.3f}")
    print(f"median us per call:  {median(microseconds):.3f}")
    print(f"worst us per call:   {max(microseconds):.3f}")


if __name__ == "__main__":
    main()
