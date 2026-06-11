from __future__ import annotations

from stark import Layout, LayoutField
from stark.algebraist.layout import AlgebraistLayoutLooped


def test_stark_layout_converts_to_algebraist_layout() -> None:
    layout = Layout(
        fields=(
            LayoutField("u", translation="du", shape=(2, 2)),
            LayoutField("v", translation="dv", shape=(2, 2)),
        )
    )

    algebraist_layout = layout.to_algebraist_layout()

    assert tuple(str(path) for path in algebraist_layout.state_paths) == ("u", "v")
    assert tuple(str(path) for path in algebraist_layout.translation_paths) == ("du", "dv")
    assert all(isinstance(field.policy, AlgebraistLayoutLooped) for field in algebraist_layout)


def test_stark_layout_defaults_translation_to_state_path() -> None:
    layout = Layout(fields=("u", "v"))
    algebraist_layout = layout.to_algebraist_layout()

    assert tuple(str(path) for path in algebraist_layout.state_paths) == ("u", "v")
    assert tuple(str(path) for path in algebraist_layout.translation_paths) == ("u", "v")


def test_stark_layout_accepts_single_string_field() -> None:
    layout = Layout("uv")
    algebraist_layout = layout.to_algebraist_layout()

    assert tuple(str(path) for path in algebraist_layout.state_paths) == ("uv",)
    assert tuple(str(path) for path in algebraist_layout.translation_paths) == ("uv",)


def test_stark_layout_accepts_field_mapping() -> None:
    layout = Layout({"u": {"translation": "du", "shape": (2, 2)}})
    algebraist_layout = layout.to_algebraist_layout()

    assert tuple(str(path) for path in algebraist_layout.state_paths) == ("u",)
    assert tuple(str(path) for path in algebraist_layout.translation_paths) == ("du",)
    assert all(isinstance(field.policy, AlgebraistLayoutLooped) for field in algebraist_layout)


def test_stark_layout_field_rejects_invalid_shape() -> None:
    try:
        LayoutField("u", shape=(2, 0)).to_algebraist_field()
    except ValueError as exc:
        assert "shape dimensions must be positive" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected invalid shape to fail.")
