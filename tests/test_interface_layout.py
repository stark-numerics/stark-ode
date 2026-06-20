from __future__ import annotations

from stark import Frame, FrameField
from stark.engines.shared.algebraist.frame import AlgebraistFrameLooped


def test_stark_layout_converts_to_algebraist_frame() -> None:
    frame = Frame(
        fields=(
            FrameField("u", translation="du", shape=(2, 2)),
            FrameField("v", translation="dv", shape=(2, 2)),
        )
    )

    algebraist_frame = frame.to_algebraist_frame()

    assert tuple(str(path) for path in algebraist_frame.state_paths) == ("u", "v")
    assert tuple(str(path) for path in algebraist_frame.translation_paths) == ("du", "dv")
    assert all(isinstance(field.policy, AlgebraistFrameLooped) for field in algebraist_frame)


def test_stark_layout_defaults_translation_to_state_path() -> None:
    frame = Frame(fields=("u", "v"))
    algebraist_frame = frame.to_algebraist_frame()

    assert tuple(str(path) for path in algebraist_frame.state_paths) == ("u", "v")
    assert tuple(str(path) for path in algebraist_frame.translation_paths) == ("u", "v")


def test_stark_layout_accepts_single_string_field() -> None:
    frame = Frame("uv")
    algebraist_frame = frame.to_algebraist_frame()

    assert tuple(str(path) for path in algebraist_frame.state_paths) == ("uv",)
    assert tuple(str(path) for path in algebraist_frame.translation_paths) == ("uv",)


def test_stark_layout_accepts_field_mapping() -> None:
    frame = Frame({"u": {"translation": "du", "shape": (2, 2)}})
    algebraist_frame = frame.to_algebraist_frame()

    assert tuple(str(path) for path in algebraist_frame.state_paths) == ("u",)
    assert tuple(str(path) for path in algebraist_frame.translation_paths) == ("du",)
    assert all(isinstance(field.policy, AlgebraistFrameLooped) for field in algebraist_frame)


def test_stark_layout_field_rejects_invalid_shape() -> None:
    try:
        FrameField("u", shape=(2, 0)).to_algebraist_field()
    except ValueError as exc:
        assert "shape dimensions must be positive" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected invalid shape to fail.")
