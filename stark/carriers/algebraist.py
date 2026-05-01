from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from stark.algebraist import Algebraist, AlgebraistField


class CarrierKernelAlgebraist:
    def __init__(
        self,
        *,
        algebraist: Algebraist | None = None,
        fields: Sequence[AlgebraistField] | None = None,
        fused_up_to: int = 12,
        generate_norm: str | None = None,
    ) -> None:
        if algebraist is None:
            if fields is None:
                raise ValueError(
                    "CarrierKernelAlgebraist requires either algebraist or fields."
                )

            algebraist = Algebraist(
                fields=fields,
                fused_up_to=fused_up_to,
                generate_norm=generate_norm,
            )

        self.algebraist = algebraist

    def bind(
        self,
        template: Any,
        carrier: Any,
        norm_policy: Any,
    ) -> "CarrierKernelAlgebraistBound":
        return CarrierKernelAlgebraistBound(
            algebraist=self.algebraist,
            norm_policy=norm_policy,
        )


@dataclass(slots=True)
class CarrierKernelAlgebraistBound:
    algebraist: Algebraist
    norm_policy: Any

    def translate(self, origin: Any, delta: Any) -> Any:
        raise NotImplementedError(
            "CarrierKernelAlgebraist does not provide returning translate(). "
            "Use in-place routing."
        )

    def add(self, left: Any, right: Any) -> Any:
        raise NotImplementedError(
            "CarrierKernelAlgebraist does not provide returning add(). "
            "Use in-place routing."
        )

    def scale(self, scalar: float, value: Any) -> Any:
        raise NotImplementedError(
            "CarrierKernelAlgebraist does not provide returning scale(). "
            "Use in-place routing."
        )

    def combine(self, coefficients: Any, values: Any) -> Any:
        raise NotImplementedError(
            "CarrierKernelAlgebraist does not provide returning combine(). "
            "Use in-place routing."
        )

    def translate_into(self, result: Any, origin: Any, delta: Any) -> None:
        self.algebraist.apply(delta, origin, result)

    def add_into(self, result: Any, left: Any, right: Any) -> None:
        self.combine_into(result, [1.0, 1.0], [left, right])

    def scale_into(self, result: Any, scalar: float, value: Any) -> None:
        self.algebraist.linear_combine[0](result, scalar, value)

    def combine_into(self, result: Any, coefficients: Any, values: Any) -> None:
        if not values:
            raise ValueError("Cannot combine empty values.")

        if len(values) > len(self.algebraist.linear_combine):
            raise ValueError(
                "CarrierKernelAlgebraist cannot combine "
                f"{len(values)} values; fused_up_to is "
                f"{len(self.algebraist.linear_combine)}."
            )

        wrapper = self.algebraist.linear_combine[len(values) - 1]
        arguments: list[Any] = [result]

        for coefficient, value in zip(coefficients, values, strict=True):
            arguments.extend([coefficient, value])

        wrapper(*arguments)

    def norm(self, value: Any) -> float:
        if self.algebraist.norm is not None:
            return self.algebraist.norm(value)

        return self.norm_policy(value)


__all__ = [
    "CarrierKernelAlgebraist",
    "CarrierKernelAlgebraistBound",
]
