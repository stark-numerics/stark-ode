from __future__ import annotations


def linear_combine_names(fused_up_to: int) -> tuple[str, ...]:
    return tuple(combine_wrapper_name(term_count) for term_count in range(1, fused_up_to + 1))


def combine_wrapper_name(term_count: int) -> str:
    return "scale" if term_count == 1 else f"combine{term_count}"


def combine_kernel_name(term_count: int) -> str:
    return "scale_kernel" if term_count == 1 else f"combine{term_count}_kernel"
