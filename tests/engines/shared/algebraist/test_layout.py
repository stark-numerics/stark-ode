from __future__ import annotations

from types import SimpleNamespace

import pytest

from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameBroadcast,
    AlgebraistField,
    AlgebraistFrameLooped,
    AlgebraistInnerProductExcluded,
    AlgebraistInnerProductL2,
    AlgebraistNormExcluded,
    AlgebraistNormRMS,
    AlgebraistFieldPath,
    AlgebraistFrameUnravel,
)


def test_layout_path_normalizes_dotted_string() -> None:
    path = AlgebraistFieldPath.from_value("position.velocity")

    assert path.parts == ("position", "velocity")
    assert path.name == "position_velocity"
    assert path.expression("state") == "state.position.velocity"


def test_layout_path_normalizes_sequence() -> None:
    path = AlgebraistFieldPath.from_value(("position", "velocity"))

    assert path.parts == ("position", "velocity")


@pytest.mark.parametrize(
    "value",
    ["", ".", "position.1x", ("position", ""), ("position", "not-valid")],
)
def test_layout_path_rejects_invalid_segments(value) -> None:
    with pytest.raises(ValueError):
        AlgebraistFieldPath.from_value(value)


def test_layout_path_rejects_invalid_expression_root() -> None:
    path = AlgebraistFieldPath.from_value("position")

    with pytest.raises(ValueError):
        path.expression("not valid")


def test_layout_path_get_set_and_ensure_traverse_runtime_objects() -> None:
    path = AlgebraistFieldPath.from_value("position.velocity")
    root = SimpleNamespace()

    parent = path.ensure_parent(root)

    assert parent is root.position

    path.assign(root, 3.0)

    assert root.position.velocity == 3.0
    assert path(root) == 3.0


def test_layout_field_normalizes_paths_and_policy() -> None:
    field = AlgebraistField(
        translation_path="delta.position",
        state_path=("state", "position"),
        policy=AlgebraistFrameLooped(shape=(3,)),
    )

    assert field.translation_path.parts == ("delta", "position")
    assert field.state_path.parts == ("state", "position")
    assert field.translation_name == "delta_position"
    assert field.state_name == "state_position"
    assert field.translation_expression("translation") == "translation.delta.position"
    assert field.state_expression("origin") == "origin.state.position"
    assert field.policy == AlgebraistFrameLooped(rank=1, shape=(3,))


def test_layout_field_defaults_to_broadcast_policy() -> None:
    field = AlgebraistField(
        translation_path="delta",
        state_path="state",
    )

    assert field.policy == AlgebraistFrameBroadcast()


def test_looped_policy_requires_rank_or_shape() -> None:
    with pytest.raises(ValueError):
        AlgebraistFrameLooped()


def test_looped_policy_rejects_rank_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        AlgebraistFrameLooped(rank=2, shape=(3,))


def test_small_fixed_policy_rejects_large_shape() -> None:
    with pytest.raises(ValueError):
        AlgebraistFrameUnravel(shape=(17,))


def test_layout_rejects_empty_field_list() -> None:
    with pytest.raises(ValueError):
        AlgebraistFrame(fields=())


def test_layout_rejects_duplicate_translation_paths() -> None:
    with pytest.raises(ValueError):
        AlgebraistFrame(
            fields=(
                AlgebraistField("delta", "state_a"),
                AlgebraistField("delta", "state_b"),
            )
        )


def test_layout_rejects_duplicate_state_paths() -> None:
    with pytest.raises(ValueError):
        AlgebraistFrame(
            fields=(
                AlgebraistField("delta_a", "state"),
                AlgebraistField("delta_b", "state"),
            )
        )


def test_layout_exposes_norm_entries_and_paths() -> None:
    first = AlgebraistField("delta_a", "state_a")
    second = AlgebraistField(
        "delta_b",
        "state_b",
    )

    frame = AlgebraistFrame(
        fields=(first, second),
        norms=(AlgebraistNormRMS(), AlgebraistNormExcluded()),
        inner_products=(AlgebraistInnerProductL2(), AlgebraistInnerProductExcluded()),
    )

    assert len(frame) == 2
    assert tuple(frame) == (first, second)
    assert frame.norm_entries == ((first, frame.norms[0]),)
    assert frame.inner_product_entries == ((first, frame.inner_products[0]),)
    assert frame.translation_paths == (
        AlgebraistFieldPath.from_value("delta_a"),
        AlgebraistFieldPath.from_value("delta_b"),
    )
    assert frame.state_paths == (
        AlgebraistFieldPath.from_value("state_a"), 
        AlgebraistFieldPath.from_value("state_b")
    )
