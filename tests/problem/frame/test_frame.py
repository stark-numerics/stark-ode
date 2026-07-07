from __future__ import annotations

from stark import Frame, Field


def test_stark_frame_accepts_explicit_fields() -> None:
    frame = Frame(
        fields=(
            Field("u", translation="du", shape=(2, 2)),
            Field("v", translation="dv", shape=(2, 2)),
        )
    )

    assert tuple(str(field.state_path) for field in frame.fields) == ("u", "v")
    assert tuple(str(field.translation_path) for field in frame.fields) == ("du", "dv")
    assert all(field.policy.kind == "looped" for field in frame.fields)
    assert all(field.shape == (2, 2) for field in frame.fields)


def test_stark_frame_defaults_translation_to_state_path() -> None:
    frame = Frame(fields=("u", "v"))

    assert tuple(str(field.state_path) for field in frame.fields) == ("u", "v")
    assert tuple(str(field.translation_path) for field in frame.fields) == ("u", "v")


def test_stark_frame_accepts_single_string_field() -> None:
    frame = Frame("uv")

    assert tuple(str(field.state_path) for field in frame.fields) == ("uv",)
    assert tuple(str(field.translation_path) for field in frame.fields) == ("uv",)


def test_stark_frame_accepts_field_mapping() -> None:
    frame = Frame({"u": {"translation": "du", "shape": (2, 2)}})

    assert tuple(str(field.state_path) for field in frame.fields) == ("u",)
    assert tuple(str(field.translation_path) for field in frame.fields) == ("du",)
    assert all(field.policy.kind == "looped" for field in frame.fields)
    assert all(field.shape == (2, 2) for field in frame.fields)


def test_stark_frame_scalar_factory_matches_mapping_syntax() -> None:
    frame = Frame.scalar("u", translation="du")
    explicit = Frame({"u": {"translation": "du", "shape": (1,)}})

    assert frame.fields == explicit.fields


def test_stark_frame_vector_factory_matches_mapping_syntax() -> None:
    frame = Frame.vector("u", translation="du", length=3)
    explicit = Frame({"u": {"translation": "du", "shape": (3,)}})

    assert frame.fields == explicit.fields


def test_stark_frame_array_factory_matches_mapping_syntax() -> None:
    frame = Frame.array("u", translation="du", shape=(4, 5))
    explicit = Frame({"u": {"translation": "du", "shape": (4, 5)}})

    assert frame.fields == explicit.fields


def test_stark_frame_field_rejects_invalid_shape() -> None:
    try:
        Field("u", shape=(2, 0))
    except ValueError as exc:
        assert "shape dimensions must be positive" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected invalid shape to fail.")
