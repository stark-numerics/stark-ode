from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Generic, cast

from stark.core.block import BlockBasis
from stark.core.block.materialize import BlockOperatorMaterialize
from stark.core.contracts import (
    Accelerator,
    BlockLike,
    BlockOperatorDiagonalLike,
    InverterOutputMode,
    InverterRequest,
    TranslationType,
)
from stark.core.contracts.engines.translation_basis import TranslationBasisLike
from stark.methods.inverters.nucleus import InverterNucleus, InverterNucleusFactor
from stark.methods.inverters.support import (
    InverterDescriptor,
    InverterRecordSolve,
    MonitorInverterLike,
    with_inverter_monitoring,
)


@dataclass(slots=True)
class InverterDenseInstance(Generic[TranslationType]):
    """Block-operator-bound dense solve action with cached compact block matrices."""

    inverter: InverterDense[TranslationType]
    matrices: list[list[float]]
    images: list[list[float]]
    results: list[list[float]]
    factors: list[InverterNucleusFactor]
    redirect_call: Callable[[BlockLike[TranslationType], BlockLike[TranslationType]], None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.redirect_call = self.call_monitored if self.inverter.monitor is not None else self.call_body

    def __call__(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.redirect_call(residual, output)

    def call_body(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        bases = self.inverter.basis.bases
        for block_index, basis in enumerate(bases):
            image = self.images[block_index]
            result = self.results[block_index]
            basis.coordinates(residual[block_index], image)
            self.factors[block_index](image, result)
            output[block_index] = basis.synthesize(result, output[block_index])

    def call_monitored(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.call_body(residual, output)
        self.inverter.record_solve(
            converged=True,
            iteration_count=None,
            initial_residual=None,
            final_residual=None,
        )


@dataclass(slots=True)
class InverterDenseInstanceSingle(Generic[TranslationType]):
    """Block-operator-bound dense solve action for a single cached block matrix."""

    inverter: InverterDense[TranslationType]
    basis: TranslationBasisLike[TranslationType]
    matrix: list[float]
    image: list[float]
    result: list[float]
    factor: InverterNucleusFactor
    redirect_call: Callable[[BlockLike[TranslationType], BlockLike[TranslationType]], None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.redirect_call = self.call_monitored if self.inverter.monitor is not None else self.call_body

    def __call__(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.redirect_call(residual, output)

    def call_body(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.basis.coordinates(residual[0], self.image)
        self.factor(self.image, self.result)
        output[0] = self.basis.synthesize(self.result, output[0])

    def call_monitored(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.call_body(residual, output)
        self.inverter.record_solve(
            converged=True,
            iteration_count=None,
            initial_residual=None,
            final_residual=None,
        )


# Optional extension: records inverter monitor events.
# Provides: record_solve.
@with_inverter_monitoring
@dataclass(slots=True)
class InverterDense(Generic[TranslationType]):
    """
    Dense direct inverter for block-diagonal small dense systems.

    Problem:
        Solve request.operator(output) = request.residual.

    Algorithm:
        1. Prepare one compact dense coordinate matrix per diagonal block.
        2. Materialise each block operator in its translation basis.
        3. Analyse each residual block into dense coordinates.
        4. Solve each compact dense system with an InverterNucleus.
        5. Synthesize each dense solution into the output block.

    Monitoring follows the same split used by schemes: unmonitored call paths do
    not invoke the monitor wrapper at all.  Monitored paths are selected once at
    preparation/instance creation time.
    """

    basis: BlockBasis[TranslationType]
    accelerator: Accelerator | None = None
    monitor: MonitorInverterLike | None = None
    materializer: BlockOperatorMaterialize[TranslationType] | None = field(init=False, default=None)
    matrices: list[list[float]] = field(init=False, default_factory=list, repr=False)
    images: list[list[float]] = field(init=False, default_factory=list, repr=False)
    results: list[list[float]] = field(init=False, default_factory=list, repr=False)
    nuclei: list[InverterNucleus] = field(init=False, default_factory=list, repr=False)
    redirect_call: Callable[[InverterRequest[TranslationType], BlockLike[TranslationType]], None] = field(init=False)

    descriptor: ClassVar[InverterDescriptor] = InverterDescriptor("Dense", "Dense direct")
    output_mode: ClassVar[InverterOutputMode] = InverterOutputMode.overwrite
    # Installed by with_inverter_monitoring from stark.methods.inverters.support.
    record_solve: ClassVar[InverterRecordSolve]

    def __post_init__(self) -> None:
        self.redirect_call = self.call_prepare

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        return self.redirect_call(request, output)

    def make_nucleus(self, dimension: int) -> InverterNucleus:
        return InverterNucleus(dimension, accelerator=self.accelerator)

    def call_prepare(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        operator = cast(BlockOperatorDiagonalLike[TranslationType], request.operator)
        source = 0.0 * output  # type: ignore[operator]
        image = 0.0 * output  # type: ignore[operator]
        self.matrices = []
        self.images = []
        self.results = []
        self.nuclei = []
        for basis in self.basis.bases:
            dimension = basis.dimension
            self.matrices.append([0.0 for _index in range(dimension * dimension)])
            self.images.append([0.0 for _index in range(dimension)])
            self.results.append([0.0 for _index in range(dimension)])
            self.nuclei.append(self.make_nucleus(dimension))

        self.materializer = BlockOperatorMaterialize(
            operator=operator,
            basis=self.basis,
            source=source,
            image=image,
            refresh_initial=False,
        )

        if len(self.basis.bases) == 1:
            self.redirect_call = (
                self.call_fast_single_monitored
                if self.monitor is not None
                else self.call_fast_single
            )
        else:
            self.redirect_call = (
                self.call_fast_monitored
                if self.monitor is not None
                else self.call_fast
            )
        return self.redirect_call(request, output)

    def call_fast_single(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        materializer = self.materializer
        operator = cast(BlockOperatorDiagonalLike[TranslationType], request.operator)
        basis = self.basis.bases[0]
        matrix = self.matrices[0]
        image = self.images[0]
        result = self.results[0]

        materializer.refresh_block_prepared(  # type: ignore[union-attr]
            0,
            operator,
            matrix,
        )
        basis.coordinates(request.residual[0], image)
        self.nuclei[0](matrix, image, result)
        output[0] = basis.synthesize(result, output[0])

    def call_fast_single_monitored(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.call_fast_single(request, output)
        self.record_solve(
            converged=True,
            iteration_count=None,
            initial_residual=None,
            final_residual=None,
        )

    def call_fast(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        materializer = self.materializer
        operator = cast(BlockOperatorDiagonalLike[TranslationType], request.operator)
        bases = self.basis.bases

        for block_index, basis in enumerate(bases):
            matrix = self.matrices[block_index]
            image = self.images[block_index]
            result = self.results[block_index]
            materializer.refresh_block_prepared(  # type: ignore[union-attr]
                block_index,
                operator,
                matrix,
            )
            basis.coordinates(request.residual[block_index], image)
            self.nuclei[block_index](matrix, image, result)
            output[block_index] = basis.synthesize(result, output[block_index])

    def call_fast_monitored(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.call_fast(request, output)
        self.record_solve(
            converged=True,
            iteration_count=None,
            initial_residual=None,
            final_residual=None,
        )

    def instance(
        self,
        operator: BlockOperatorDiagonalLike[TranslationType],
    ) -> InverterDenseInstance[TranslationType] | InverterDenseInstanceSingle[TranslationType]:
        matrices: list[list[float]] = []
        images: list[list[float]] = []
        results: list[list[float]] = []
        factors: list[InverterNucleusFactor] = []

        for block_index, basis in enumerate(self.basis.bases):
            dimension = basis.dimension
            matrix = [0.0 for _index in range(dimension * dimension)]
            dense_fill = getattr(operator[block_index], "dense_fill", None)
            if not callable(dense_fill):
                raise TypeError("Block-operator-bound dense inverter instances require dense_fill entries.")
            dense_fill(basis, matrix, 0, 0, dimension)
            matrices.append(matrix)
            images.append([0.0 for _index in range(dimension)])
            results.append([0.0 for _index in range(dimension)])
            factors.append(self.make_nucleus(dimension).factor(matrix))

        if len(self.basis.bases) == 1:
            return InverterDenseInstanceSingle(
                inverter=self,
                basis=self.basis.bases[0],
                matrix=matrices[0],
                image=images[0],
                result=results[0],
                factor=factors[0],
            )

        return InverterDenseInstance(
            inverter=self,
            matrices=matrices,
            images=images,
            results=results,
            factors=factors,
        )


__all__ = ["InverterDense", "InverterDenseInstance", "InverterDenseInstanceSingle"]
