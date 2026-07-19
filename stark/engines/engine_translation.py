from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, SupportsFloat, cast

from stark.core.contracts.engines.allocator import AllocatorLike
from stark.core.contracts.engines.carrier import CarrierLike
from stark.core.contracts.problem.frame import FrameLike


EngineTranslationApplyField = Callable[[object, object], None]
EngineTranslationAddField = Callable[["EngineTranslation", "EngineTranslation"], None]
EngineTranslationScaleField = Callable[[float, "EngineTranslation"], None]


@dataclass
class EngineTranslation:
    """Engine-owned `Translation` for frame-backed states.

    The problem `Frame` describes which state and translation fields exist; the
    engine supplies carriers that know how those fields are stored and updated.
    `EngineTranslation` is the small concrete object that joins those two
    pieces into the core `Translation` contract.

    It deliberately chooses its field update strategy during construction.
    Return-style carriers such as JAX assign the value returned by carrier
    arithmetic, while into-style carriers such as NumPy, CuPy, and native
    arrays let the carrier mutate the supplied result value.
    """

    frame: FrameLike = field(repr=False)
    carriers: tuple[CarrierLike[Any, Any], ...] = field(repr=False)
    allocator: AllocatorLike[Any, EngineTranslation] = field(repr=False)
    linear_combine: tuple[Callable[..., EngineTranslation], ...] = field(
        default=(),
        repr=False,
    )
    apply_translation: Callable[[object, EngineTranslation, object], object] | None = (
        field(default=None, repr=False)
    )
    norm_kernel: Callable[[EngineTranslation], float] | None = field(
        default=None,
        repr=False,
    )
    apply_translation_fields: Callable[[object, object], None] = field(
        init=False,
        repr=False,
    )
    add_fields: Callable[[EngineTranslation, EngineTranslation], None] = field(
        init=False,
        repr=False,
    )
    scale_fields: Callable[[float, EngineTranslation], None] = field(
        init=False,
        repr=False,
    )
    apply_translation_field_ops: tuple[EngineTranslationApplyField, ...] = field(
        init=False,
        repr=False,
        default=(),
    )
    add_field_ops: tuple[EngineTranslationAddField, ...] = field(
        init=False,
        repr=False,
        default=(),
    )
    scale_field_ops: tuple[EngineTranslationScaleField, ...] = field(
        init=False,
        repr=False,
        default=(),
    )

    def __post_init__(self) -> None:
        preferences = {carrier.arithmetic.preference for carrier in self.carriers}
        if preferences == {"return"}:
            self.apply_translation_fields = self.apply_translation_fields_return
            self.add_fields = self.add_fields_return
            self.scale_fields = self.scale_fields_return
            return

        if preferences <= {"into"}:
            self.apply_translation_fields = self.apply_translation_fields_into
            self.add_fields = self.add_fields_into
            self.scale_fields = self.scale_fields_into
            return

        self.apply_translation_field_ops = self.build_apply_translation_field_ops()
        self.add_field_ops = self.build_add_field_ops()
        self.scale_field_ops = self.build_scale_field_ops()
        self.apply_translation_fields = self.apply_translation_fields_mixed
        self.add_fields = self.add_fields_mixed
        self.scale_fields = self.scale_fields_mixed

    def __call__(self, origin: object, result: object) -> None:
        if self.apply_translation is not None:
            self.apply_translation(origin, self, result)
            return

        self.apply_translation_fields(origin, result)

    def __add__(self, other: EngineTranslation) -> EngineTranslation:
        if self.allocator is not other.allocator:
            raise ValueError("Cannot add translations allocated by different engines.")

        result = self.allocator.allocate_translation()
        self.add_fields(other, result)
        return result

    def __rmul__(self, scalar: float) -> EngineTranslation:
        result = self.allocator.allocate_translation()
        self.scale_fields(scalar, result)
        return result

    def __mul__(self, scalar: float) -> EngineTranslation:
        return self.__rmul__(scalar)

    def norm(self) -> float:
        if self.norm_kernel is not None:
            value = self.norm_kernel(self)
            item = getattr(value, "item", None)
            if callable(item):
                value = item()
            return float(cast(SupportsFloat, value))

        total = 0.0
        for field, norm, carrier in zip(
            self.frame.fields,
            self.frame.norms,
            self.carriers,
            strict=True,
        ):
            if norm.kind == "excluded":
                continue
            field_norm = carrier.norm(field.translation_path(self))
            total += field_norm * field_norm
        return sqrt(total)

    def apply_translation_fields_into(self, origin: object, result: object) -> None:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            carrier.arithmetic.translate(
                field.state_path(origin),
                1.0,
                field.translation_path(self),
                field.state_path(result),
            )

    def apply_translation_fields_return(self, origin: object, result: object) -> None:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            field.state_path.assign(
                result,
                carrier.arithmetic.translate(
                    field.state_path(origin),
                    1.0,
                    field.translation_path(self),
                    field.state_path(result),
                ),
            )

    def apply_translation_fields_mixed(self, origin: object, result: object) -> None:
        for operation in self.apply_translation_field_ops:
            operation(origin, result)

    def add_fields_into(
        self,
        other: EngineTranslation,
        result: EngineTranslation,
    ) -> None:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            carrier.arithmetic.add(
                field.translation_path(self),
                field.translation_path(other),
                field.translation_path(result),
            )

    def add_fields_return(
        self,
        other: EngineTranslation,
        result: EngineTranslation,
    ) -> None:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            field.translation_path.assign(
                result,
                carrier.arithmetic.add(
                    field.translation_path(self),
                    field.translation_path(other),
                    field.translation_path(result),
                ),
            )

    def add_fields_mixed(
        self,
        other: EngineTranslation,
        result: EngineTranslation,
    ) -> None:
        for operation in self.add_field_ops:
            operation(other, result)

    def scale_fields_into(self, scalar: float, result: EngineTranslation) -> None:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            carrier.arithmetic.scale(
                scalar,
                field.translation_path(self),
                field.translation_path(result),
            )

    def scale_fields_return(self, scalar: float, result: EngineTranslation) -> None:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            field.translation_path.assign(
                result,
                carrier.arithmetic.scale(
                    scalar,
                    field.translation_path(self),
                    field.translation_path(result),
                ),
            )

    def scale_fields_mixed(self, scalar: float, result: EngineTranslation) -> None:
        for operation in self.scale_field_ops:
            operation(scalar, result)

    def build_apply_translation_field_ops(
        self,
    ) -> tuple[EngineTranslationApplyField, ...]:
        operations: list[EngineTranslationApplyField] = []
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            if carrier.arithmetic.preference == "return":

                def operation(
                    origin: object,
                    result: object,
                    *,
                    field: Any = field,
                    carrier: CarrierLike[Any, Any] = carrier,
                ) -> None:
                    field.state_path.assign(
                        result,
                        carrier.arithmetic.translate(
                            field.state_path(origin),
                            1.0,
                            field.translation_path(self),
                            field.state_path(result),
                        ),
                    )

            else:

                def operation(
                    origin: object,
                    result: object,
                    *,
                    field: Any = field,
                    carrier: CarrierLike[Any, Any] = carrier,
                ) -> None:
                    carrier.arithmetic.translate(
                        field.state_path(origin),
                        1.0,
                        field.translation_path(self),
                        field.state_path(result),
                    )

            operations.append(operation)
        return tuple(operations)

    def build_add_field_ops(self) -> tuple[EngineTranslationAddField, ...]:
        operations: list[EngineTranslationAddField] = []
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            if carrier.arithmetic.preference == "return":

                def operation(
                    other: EngineTranslation,
                    result: EngineTranslation,
                    *,
                    field: Any = field,
                    carrier: CarrierLike[Any, Any] = carrier,
                ) -> None:
                    field.translation_path.assign(
                        result,
                        carrier.arithmetic.add(
                            field.translation_path(self),
                            field.translation_path(other),
                            field.translation_path(result),
                        ),
                    )

            else:

                def operation(
                    other: EngineTranslation,
                    result: EngineTranslation,
                    *,
                    field: Any = field,
                    carrier: CarrierLike[Any, Any] = carrier,
                ) -> None:
                    carrier.arithmetic.add(
                        field.translation_path(self),
                        field.translation_path(other),
                        field.translation_path(result),
                    )

            operations.append(operation)
        return tuple(operations)

    def build_scale_field_ops(self) -> tuple[EngineTranslationScaleField, ...]:
        operations: list[EngineTranslationScaleField] = []
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            if carrier.arithmetic.preference == "return":

                def operation(
                    scalar: float,
                    result: EngineTranslation,
                    *,
                    field: Any = field,
                    carrier: CarrierLike[Any, Any] = carrier,
                ) -> None:
                    field.translation_path.assign(
                        result,
                        carrier.arithmetic.scale(
                            scalar,
                            field.translation_path(self),
                            field.translation_path(result),
                        ),
                    )

            else:

                def operation(
                    scalar: float,
                    result: EngineTranslation,
                    *,
                    field: Any = field,
                    carrier: CarrierLike[Any, Any] = carrier,
                ) -> None:
                    carrier.arithmetic.scale(
                        scalar,
                        field.translation_path(self),
                        field.translation_path(result),
                    )

            operations.append(operation)
        return tuple(operations)


__all__ = ["EngineTranslation"]
