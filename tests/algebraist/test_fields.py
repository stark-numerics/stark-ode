from dataclasses import dataclass

import numpy as np
import pytest

from stark.algebraist import (
    Algebraist,
    AlgebraistBroadcast,
    AlgebraistField,
    AlgebraistLooped,
    AlgebraistSmallFixed,
)


@dataclass
class Payload:
    value: np.ndarray


@dataclass
class NestedBox:
    payload: Payload


def test_field_normalizes_dotted_paths():
    field = AlgebraistField("delta.value", "state.value")

    assert field.translation_path == ("delta", "value")
    assert field.state_path == ("state", "value")
    assert field.translation_name == "delta_value"
    assert field.state_name == "state_value"


def test_field_rejects_invalid_path_segment():
    with pytest.raises(ValueError, match="valid Python identifier"):
        AlgebraistField("delta.0value", "state.value")


def test_looped_policy_requires_rank_or_shape():
    with pytest.raises(ValueError, match="rank or shape"):
        AlgebraistField("value", "value", policy=AlgebraistLooped())


def test_looped_policy_infers_rank_from_shape():
    field = AlgebraistField("value", "value", policy=AlgebraistLooped(shape=(2, 3)))

    assert field.policy == AlgebraistLooped(rank=2, shape=(2, 3))


def test_small_fixed_policy_rejects_large_shape():
    with pytest.raises(ValueError, match="too large"):
        AlgebraistField("value", "value", policy=AlgebraistSmallFixed(shape=(17,)))


def test_algebraist_exposes_linear_combine_through_combine12_by_default():
    algebraist = Algebraist(fields=[AlgebraistField("value", "value")])

    assert len(algebraist.linear_combine) == 12
    assert "combine12" in algebraist.wrapper_sources
    assert not hasattr(algebraist, "linear_combination")


def test_nested_wrapper_source_uses_nested_paths():
    algebraist = Algebraist(
        fields=[
            AlgebraistField(
                translation_path=("payload", "value"),
                state_path=("payload", "value"),
                policy=AlgebraistBroadcast(),
            )
        ],
        generate_norm="rms",
    )

    assert "out.payload.value" in algebraist.wrapper_sources["scale"]
    assert "translation.payload.value" in algebraist.wrapper_sources["apply"]
    assert "origin.payload.value" in algebraist.wrapper_sources["apply"]
    assert "translation.payload.value" in algebraist.wrapper_sources["norm"]


def test_nested_field_operations_update_nested_payloads():
    algebraist = Algebraist(
        fields=[
            AlgebraistField(
                translation_path=("payload", "value"),
                state_path=("payload", "value"),
                policy=AlgebraistBroadcast(),
            )
        ],
        generate_norm="rms",
    )

    origin = NestedBox(Payload(np.array([1.0, 2.0])))
    delta = NestedBox(Payload(np.array([3.0, 4.0])))
    result = NestedBox(Payload(np.zeros(2)))

    algebraist.apply(delta, origin, result)
    np.testing.assert_allclose(result.payload.value, np.array([4.0, 6.0]))

    algebraist.linear_combine[0](result, 2.0, delta)
    np.testing.assert_allclose(result.payload.value, np.array([6.0, 8.0]))

    assert algebraist.norm(delta) == pytest.approx((12.5) ** 0.5)


def test_generated_linear_combine_matches_expected_array_results():
    algebraist = Algebraist(fields=[AlgebraistField("value", "value")])
    out = Payload(np.zeros(2))
    values = [
        Payload(np.array([float(index), float(index + 1)]))
        for index in range(1, 13)
    ]

    scaled = algebraist.linear_combine[0](out, 2.0, values[0])
    np.testing.assert_allclose(scaled.value, np.array([2.0, 4.0]))

    combined2 = algebraist.linear_combine[1](out, 2.0, values[0], 3.0, values[1])
    np.testing.assert_allclose(combined2.value, np.array([8.0, 13.0]))

    terms = []
    expected = np.zeros(2)
    for index, value in enumerate(values, start=1):
        coefficient = float(index)
        terms.extend([coefficient, value])
        expected += coefficient * value.value

    combined12 = algebraist.linear_combine[11](out, *terms)

    assert combined12 is out
    np.testing.assert_allclose(out.value, expected)
