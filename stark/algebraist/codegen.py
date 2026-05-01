from __future__ import annotations

from collections.abc import Sequence
from math import prod

from stark.algebraist.fields import AlgebraistField
from stark.algebraist.policies import AlgebraistBroadcast, AlgebraistLooped, AlgebraistSmallFixed


def field_combine_assignment(field: AlgebraistField, term_count: int) -> str:
    policy = field.policy
    if isinstance(policy, AlgebraistSmallFixed):
        return small_fixed_combine_assignment(field, policy.shape, term_count)
    if isinstance(policy, AlgebraistLooped):
        assert policy.rank is not None
        return looped_combine_assignment(field, policy.rank, term_count)
    if isinstance(policy, AlgebraistBroadcast):
        return broadcast_combine_assignment(field, term_count)
    raise TypeError(f"Unknown Algebraist policy {policy!r}.")


def field_tableau_combine_assignment(
    field: AlgebraistField,
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    policy = field.policy
    if isinstance(policy, AlgebraistSmallFixed):
        return small_fixed_tableau_combine_assignment(field, policy.shape, coefficients, term_count)
    if isinstance(policy, AlgebraistLooped):
        assert policy.rank is not None
        return looped_tableau_combine_assignment(field, policy.rank, coefficients, term_count)
    if isinstance(policy, AlgebraistBroadcast):
        return broadcast_tableau_combine_assignment(field, coefficients, term_count)
    raise TypeError(f"Unknown Algebraist policy {policy!r}.")


def field_tableau_stage_assignment(
    field: AlgebraistField,
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    policy = field.policy
    if isinstance(policy, AlgebraistSmallFixed):
        return small_fixed_tableau_stage_assignment(field, policy.shape, coefficients, term_count)
    if isinstance(policy, AlgebraistLooped):
        assert policy.rank is not None
        return looped_tableau_stage_assignment(field, policy.rank, coefficients, term_count)
    if isinstance(policy, AlgebraistBroadcast):
        return broadcast_tableau_stage_assignment(field, coefficients, term_count)
    raise TypeError(f"Unknown Algebraist policy {policy!r}.")


def field_apply_assignment(field: AlgebraistField) -> str:
    policy = field.policy
    if isinstance(policy, AlgebraistSmallFixed):
        return small_fixed_apply_assignment(field, policy.shape)
    if isinstance(policy, AlgebraistLooped):
        assert policy.rank is not None
        return looped_apply_assignment(field, policy.rank)
    if isinstance(policy, AlgebraistBroadcast):
        return broadcast_apply_assignment(field)
    raise TypeError(f"Unknown Algebraist policy {policy!r}.")


def broadcast_combine_assignment(field: AlgebraistField, term_count: int) -> str:
    name = field.translation_name
    value = " + ".join(f"a{index} * x{index}_{name}" for index in range(term_count))
    return f"    out_{name}[...] = {value}"


def broadcast_tableau_combine_assignment(
    field: AlgebraistField,
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    name = field.translation_name
    value = " + ".join(f"(step * {coefficients[index]!r}) * x{index}_{name}" for index in range(term_count))
    return f"    out_{name}[...] = {value}"


def broadcast_tableau_stage_assignment(
    field: AlgebraistField,
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    translation = field.translation_name
    state = field.state_name
    value = " + ".join(f"(step * {coefficients[index]!r}) * x{index}_{translation}" for index in range(term_count))
    return f"    result_{state}[...] = origin_{state} + ({value})"


def looped_combine_assignment(field: AlgebraistField, rank: int, term_count: int) -> str:
    name = field.translation_name
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        shape_binding = f"    {shape_terms[0]} = out_{name}.shape[0]"
    else:
        shape_binding = f"    {', '.join(shape_terms)} = out_{name}.shape"

    lines = [shape_binding]
    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    value = " + ".join(f"a{index} * x{index}_{name}[{location}]" for index in range(term_count))
    lines.append(f"{indent}out_{name}[{location}] = {value}")
    return "\n".join(lines)


def looped_tableau_combine_assignment(
    field: AlgebraistField,
    rank: int,
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    name = field.translation_name
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        shape_binding = f"    {shape_terms[0]} = out_{name}.shape[0]"
    else:
        shape_binding = f"    {', '.join(shape_terms)} = out_{name}.shape"

    lines = [shape_binding]
    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    value = " + ".join(
        f"(step * {coefficients[index]!r}) * x{index}_{name}[{location}]"
        for index in range(term_count)
    )
    lines.append(f"{indent}out_{name}[{location}] = {value}")
    return "\n".join(lines)


def looped_tableau_stage_assignment(
    field: AlgebraistField,
    rank: int,
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    translation = field.translation_name
    state = field.state_name
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        shape_binding = f"    {shape_terms[0]} = result_{state}.shape[0]"
    else:
        shape_binding = f"    {', '.join(shape_terms)} = result_{state}.shape"

    lines = [shape_binding]
    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    value = " + ".join(
        f"(step * {coefficients[index]!r}) * x{index}_{translation}[{location}]"
        for index in range(term_count)
    )
    lines.append(f"{indent}result_{state}[{location}] = origin_{state}[{location}] + ({value})")
    return "\n".join(lines)


def small_fixed_combine_assignment(field: AlgebraistField, shape: tuple[int, ...], term_count: int) -> str:
    name = field.translation_name
    lines: list[str] = []
    for location in fixed_locations(shape):
        value = " + ".join(f"a{index} * x{index}_{name}[{location}]" for index in range(term_count))
        lines.append(f"    out_{name}[{location}] = {value}")
    return "\n".join(lines)


def small_fixed_tableau_stage_assignment(
    field: AlgebraistField,
    shape: tuple[int, ...],
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    translation = field.translation_name
    state = field.state_name
    lines: list[str] = []
    for location in fixed_locations(shape):
        value = " + ".join(
            f"(step * {coefficients[index]!r}) * x{index}_{translation}[{location}]"
            for index in range(term_count)
        )
        lines.append(f"    result_{state}[{location}] = origin_{state}[{location}] + ({value})")
    return "\n".join(lines)


def small_fixed_tableau_combine_assignment(
    field: AlgebraistField,
    shape: tuple[int, ...],
    coefficients: Sequence[float],
    term_count: int,
) -> str:
    name = field.translation_name
    lines: list[str] = []
    for location in fixed_locations(shape):
        value = " + ".join(
            f"(step * {coefficients[index]!r}) * x{index}_{name}[{location}]"
            for index in range(term_count)
        )
        lines.append(f"    out_{name}[{location}] = {value}")
    return "\n".join(lines)


def broadcast_apply_assignment(field: AlgebraistField) -> str:
    delta = field.translation_name
    target = field.state_name
    return f"    result_{target}[...] = origin_{target} + delta_{delta}"


def looped_apply_assignment(field: AlgebraistField, rank: int) -> str:
    delta = field.translation_name
    target = field.state_name
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        shape_binding = f"    {shape_terms[0]} = delta_{delta}.shape[0]"
    else:
        shape_binding = f"    {', '.join(shape_terms)} = delta_{delta}.shape"

    lines = [shape_binding]
    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    lines.append(f"{indent}result_{target}[{location}] = origin_{target}[{location}] + delta_{delta}[{location}]")
    return "\n".join(lines)


def small_fixed_apply_assignment(field: AlgebraistField, shape: tuple[int, ...]) -> str:
    delta = field.translation_name
    target = field.state_name
    return "\n".join(
        f"    result_{target}[{location}] = origin_{target}[{location}] + delta_{delta}[{location}]"
        for location in fixed_locations(shape)
    )


def field_norm_body(fields: Sequence[AlgebraistField], kind: str) -> list[str]:
    lines = ["    total = 0.0"]
    if kind == "rms":
        lines.append("    count = 0")

    for field in fields:
        if not field.include_in_norm:
            continue
        policy = field.policy
        if isinstance(policy, AlgebraistSmallFixed):
            lines.extend(small_fixed_norm_body(field, policy.shape, kind))
        elif isinstance(policy, AlgebraistLooped):
            assert policy.rank is not None
            lines.extend(looped_norm_body(field, policy.rank, kind))
        else:
            lines.extend(broadcast_norm_body(field, kind))

    if kind == "rms":
        lines.append("    return 0.0 if count == 0 else (total / count) ** 0.5")
    else:
        lines.append("    return total ** 0.5")
    return lines


def broadcast_norm_body(field: AlgebraistField, kind: str) -> list[str]:
    name = field.translation_name
    lines = [f"    for value in {name}.ravel():", "        total += value * value"]
    if kind == "rms":
        lines.append(f"    count += {name}.size")
    return lines


def looped_norm_body(field: AlgebraistField, rank: int, kind: str) -> list[str]:
    name = field.translation_name
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        lines = [f"    {shape_terms[0]} = {name}.shape[0]"]
    else:
        lines = [f"    {', '.join(shape_terms)} = {name}.shape"]

    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    lines.append(f"{indent}value = {name}[{location}]")
    lines.append(f"{indent}total += value * value")
    if kind == "rms":
        lines.append(f"    count += {name}.size")
    return lines


def small_fixed_norm_body(field: AlgebraistField, shape: tuple[int, ...], kind: str) -> list[str]:
    name = field.translation_name
    lines: list[str] = []
    for location in fixed_locations(shape):
        lines.append(f"    value = {name}[{location}]")
        lines.append("    total += value * value")
    if kind == "rms":
        lines.append(f"    count += {prod(shape)}")
    return lines


def fixed_locations(shape: tuple[int, ...]) -> tuple[str, ...]:
    locations: list[str] = []

    def visit(prefix: tuple[int, ...], depth: int) -> None:
        if depth == len(shape):
            if len(prefix) == 1:
                locations.append(str(prefix[0]))
            else:
                locations.append(", ".join(str(item) for item in prefix))
            return
        for index in range(shape[depth]):
            visit(prefix + (index,), depth + 1)

    visit((), 0)
    return tuple(locations)
