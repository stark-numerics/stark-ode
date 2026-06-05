from __future__ import annotations

from types import SimpleNamespace

import pytest

from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutBroadcast,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
    AlgebraistLayoutPath,
    AlgebraistLayoutUnravel,
)


def test_layout_path_normalizes_dotted_string() -> None:
    path = AlgebraistLayoutPath.from_value("position.velocity")

    assert path.parts == ("position", "velocity")
    assert path.name == "position_velocity"
    assert path.expression("state") == "state.position.velocity"


def test_layout_path_normalizes_sequence() -> None:
    path = AlgebraistLayoutPath.from_value(("position", "velocity"))

    assert path.parts == ("position", "velocity")


@pytest.mark.parametrize(
    "value",
    ["", ".", "position.1x", ("position", ""), ("position", "not-valid")],
)
def test_layout_path_rejects_invalid_segments(value) -> None:
    with pytest.raises(ValueError):
        AlgebraistLayoutPath.from_value(value)


def test_layout_path_rejects_invalid_expression_root() -> None:
    path = AlgebraistLayoutPath.from_value("position")

    with pytest.raises(ValueError):
        path.expression("not valid")


def test_layout_path_get_set_and_ensure_traverse_runtime_objects() -> None:
    path = AlgebraistLayoutPath.from_value("position.velocity")
    root = SimpleNamespace()

    parent = path.ensure(root)

    assert parent is root.position

    path.set(root, 3.0)

    assert root.position.velocity == 3.0
    assert path.get(root) == 3.0


def test_layout_field_normalizes_paths_and_policy() -> None:
    field = AlgebraistLayoutField(
        translation_path="delta.position",
        state_path=("state", "position"),
        policy=AlgebraistLayoutLooped(shape=(3,)),
    )

    assert field.translation_path.parts == ("delta", "position")
    assert field.state_path.parts == ("state", "position")
    assert field.translation_name == "delta_position"
    assert field.state_name == "state_position"
    assert field.translation_expression("translation") == "translation.delta.position"
    assert field.state_expression("origin") == "origin.state.position"
    assert field.policy == AlgebraistLayoutLooped(rank=1, shape=(3,))


def test_layout_field_defaults_to_broadcast_policy() -> None:
    field = AlgebraistLayoutField(
        translation_path="delta",
        state_path="state",
    )

    assert field.policy == AlgebraistLayoutBroadcast()


def test_looped_policy_requires_rank_or_shape() -> None:
    with pytest.raises(ValueError):
        AlgebraistLayoutLooped().normalized()


def test_looped_policy_rejects_rank_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        AlgebraistLayoutLooped(rank=2, shape=(3,)).normalized()


def test_small_fixed_policy_rejects_large_shape() -> None:
    with pytest.raises(ValueError):
        AlgebraistLayoutUnravel(shape=(17,)).normalized()


def test_layout_rejects_empty_field_list() -> None:
    with pytest.raises(ValueError):
        AlgebraistLayout(fields=())


def test_layout_rejects_duplicate_translation_paths() -> None:
    with pytest.raises(ValueError):
        AlgebraistLayout(
            fields=(
                AlgebraistLayoutField("delta", "state_a"),
                AlgebraistLayoutField("delta", "state_b"),
            )
        )


def test_layout_rejects_duplicate_state_paths() -> None:
    with pytest.raises(ValueError):
        AlgebraistLayout(
            fields=(
                AlgebraistLayoutField("delta_a", "state"),
                AlgebraistLayoutField("delta_b", "state"),
            )
        )


def test_layout_exposes_norm_fields_and_paths() -> None:
    first = AlgebraistLayoutField("delta_a", "state_a", include_in_norm=True)
    second = AlgebraistLayoutField("delta_b", "state_b", include_in_norm=False)

    layout = AlgebraistLayout(fields=(first, second))

    assert len(layout) == 2
    assert tuple(layout) == (first, second)
    assert layout.norm_fields == (first,)
    assert layout.translation_paths == (
        AlgebraistLayoutPath.from_value("delta_a"),
        AlgebraistLayoutPath.from_value("delta_b"),
    )
    assert layout.state_paths == (
        AlgebraistLayoutPath.from_value("state_a"), 
        AlgebraistLayoutPath.from_value("state_b")
    )
