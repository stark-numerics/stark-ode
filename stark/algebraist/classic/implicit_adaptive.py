from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from stark.algebraist.implicit_fixed import (
    AlgebraistImplicitCombination,
    AlgebraistImplicitFixedSchemeBinder,
)


@dataclass(frozen=True, slots=True)
class AlgebraistImplicitAdaptiveSchemeBinding:
    known_shift_calls: tuple[Callable[..., object] | None, ...]
    high_delta_call: Callable[..., object] | None
    low_delta_call: Callable[..., object] | None
    error_delta_call: Callable[..., object] | None
    known_shifts: tuple[AlgebraistImplicitCombination | None, ...]
    high_delta: AlgebraistImplicitCombination | None
    low_delta: AlgebraistImplicitCombination | None
    error_delta: AlgebraistImplicitCombination | None

    def require_known_shift_call(
        self,
        stage_index: int,
        scheme_name: str,
    ) -> Callable[..., object]:
        if stage_index < 0:
            raise ValueError(
                f"{scheme_name} requested invalid generated known-shift call "
                f"{stage_index}."
            )

        try:
            known_shift_call = self.known_shift_calls[stage_index]
        except IndexError as exc:
            raise ValueError(
                f"{scheme_name} requires generated known-shift call "
                f"{stage_index}, but only {len(self.known_shift_calls)} calls "
                "were bound."
            ) from exc

        if known_shift_call is None:
            raise ValueError(
                f"{scheme_name} requires generated known-shift call "
                f"{stage_index}, but that stage has no generated algebra."
            )

        return known_shift_call

    def require_high_delta_call(self, scheme_name: str) -> Callable[..., object]:
        high_delta_call = self.high_delta_call
        if high_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated high-delta call.")

        return high_delta_call

    def require_low_delta_call(self, scheme_name: str) -> Callable[..., object]:
        low_delta_call = self.low_delta_call
        if low_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated low-delta call.")

        return low_delta_call

    def require_error_delta_call(self, scheme_name: str) -> Callable[..., object]:
        error_delta_call = self.error_delta_call
        if error_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated error-delta call.")

        return error_delta_call


@dataclass(frozen=True, slots=True)
class AlgebraistImplicitAdaptiveSchemeBinder:
    algebraist: object
    fixed_binder: AlgebraistImplicitFixedSchemeBinder | None = None

    def __call__(
        self,
        *,
        known_shifts: Sequence[AlgebraistImplicitCombination | None] = (),
        high_delta: AlgebraistImplicitCombination | None = None,
        low_delta: AlgebraistImplicitCombination | None = None,
        error_delta: AlgebraistImplicitCombination | None = None,
    ) -> AlgebraistImplicitAdaptiveSchemeBinding:
        fixed_binder = self.fixed_binder
        if fixed_binder is None:
            fixed_binder = AlgebraistImplicitFixedSchemeBinder(self.algebraist)

        known_shift_combinations = tuple(known_shifts)
        known_shift_binding = fixed_binder(known_shifts=known_shift_combinations)

        return AlgebraistImplicitAdaptiveSchemeBinding(
            known_shift_calls=known_shift_binding.known_shift_calls,
            high_delta_call=(
                None if high_delta is None else fixed_binder.optional_combination(high_delta)
            ),
            low_delta_call=(
                None if low_delta is None else fixed_binder.optional_combination(low_delta)
            ),
            error_delta_call=(
                None if error_delta is None else fixed_binder.optional_combination(error_delta)
            ),
            known_shifts=known_shift_combinations,
            high_delta=high_delta,
            low_delta=low_delta,
            error_delta=error_delta,
        )


__all__ = [
    "AlgebraistImplicitAdaptiveSchemeBinder",
    "AlgebraistImplicitAdaptiveSchemeBinding",
]
