from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from stark.core.contracts import BlockLike, BlockOperatorLike, TranslationType
from stark.methods.inverters.krylov.projection import InverterKrylovProjection

if TYPE_CHECKING:  # pragma: no cover
    from stark.methods.inverters.krylov.arnoldi import InverterKrylovArnoldi


@dataclass(slots=True)
class InverterKrylovBasis(Generic[TranslationType]):
    """Reusable Arnoldi basis for an `InverterKrylovArnoldi` window."""

    inverter: InverterKrylovArnoldi[TranslationType]
    restart: int
    size: int = -1
    vectors: list[BlockLike[TranslationType]] = field(default_factory=list)
    image: BlockLike[TranslationType] | None = None
    temporary: BlockLike[TranslationType] | None = None

    def prepare(self, size: int) -> None:
        if self.size == size:
            return
        self.size = size
        self.vectors = [
            self.inverter.allocate_block(size)
            for _index in range(self.restart + 1)
        ]
        self.image = self.inverter.allocate_block(size)
        self.temporary = self.inverter.allocate_block(size)

    def start(self, residual: BlockLike[TranslationType], beta: float) -> None:
        self.inverter.scale_block(1.0 / beta, residual, self.vectors[0])

    def build_column(
        self,
        column: int,
        operator: BlockOperatorLike[TranslationType],
        projection: InverterKrylovProjection,
        *,
        breakdown_tolerance: float,
    ) -> bool:
        """Extend the Arnoldi basis by one column.

        Returns true when the new candidate vector has broken down to zero.
        """

        image = self.image
        temporary = self.temporary
        assert image is not None
        assert temporary is not None

        operator(self.vectors[column], image)
        self.inverter.precondition_block(operator, image, temporary)
        self.inverter.copy_block(temporary, image)
        for row in range(column + 1):
            value = self.inverter.inner_product_block(image, self.vectors[row])
            projection.hessenberg[row][column] = value
            self.inverter.combine2_block(
                1.0,
                image,
                -value,
                self.vectors[row],
                temporary,
            )
            self.inverter.copy_block(temporary, image)

        norm = self.inverter.norm_block(image)
        projection.hessenberg[column + 1][column] = norm
        projection.apply_previous(column)
        if norm <= breakdown_tolerance:
            return True

        self.inverter.scale_block(1.0 / norm, image, self.vectors[column + 1])
        return False


__all__ = ["InverterKrylovBasis"]
