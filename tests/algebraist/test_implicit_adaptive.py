import numpy as np
import pytest

from stark.algebraist import (
    Algebraist,
    AlgebraistField,
    AlgebraistImplicitAdaptiveSchemeBinding,
    AlgebraistImplicitCombination,
)


class FakeTranslation:
    def __init__(self, value):
        self.value = value


def make_algebraist() -> Algebraist:
    return Algebraist(fields=(AlgebraistField("value", "value"),))


def test_implicit_adaptive_binding_generates_stage_high_low_and_error_calls() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_implicit_adaptive_scheme(
        known_shifts=(
            None,
            AlgebraistImplicitCombination("stage1_known_shift", (0.5,), (0,)),
        ),
        high_delta=AlgebraistImplicitCombination("high_delta", (0.25, 0.75), (0, 1)),
        low_delta=AlgebraistImplicitCombination("low_delta", (0.5, 0.5), (0, 1)),
        error_delta=AlgebraistImplicitCombination("error_delta", (-0.25, 0.25), (0, 1)),
    )

    assert isinstance(binding, AlgebraistImplicitAdaptiveSchemeBinding)
    assert binding.known_shift_calls[0] is None

    out = FakeTranslation(np.zeros(2))
    k0 = FakeTranslation(np.array([2.0, 4.0]))
    k1 = FakeTranslation(np.array([6.0, 8.0]))

    binding.require_known_shift_call(1, "TestESDIRK")(out, k0)
    np.testing.assert_allclose(out.value, np.array([1.0, 2.0]))

    binding.require_high_delta_call("TestESDIRK")(out, k0, k1)
    np.testing.assert_allclose(out.value, np.array([5.0, 7.0]))

    binding.require_low_delta_call("TestESDIRK")(out, k0, k1)
    np.testing.assert_allclose(out.value, np.array([4.0, 6.0]))

    binding.require_error_delta_call("TestESDIRK")(out, k0, k1)
    np.testing.assert_allclose(out.value, np.array([1.0, 1.0]))

    assert "stage1_known_shift_combine" in algebraist.sources
    assert "high_delta_combine" in algebraist.sources
    assert "low_delta_combine" in algebraist.sources
    assert "error_delta_combine" in algebraist.sources


def test_implicit_adaptive_binding_supports_step_scaled_known_shift() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_implicit_adaptive_scheme(
        known_shifts=(
            AlgebraistImplicitCombination(
                "stage1_known_shift",
                (0.5,),
                step_scale=True,
            ),
        ),
    )
    out = FakeTranslation(np.zeros(2))
    k0 = FakeTranslation(np.array([2.0, 4.0]))

    binding.require_known_shift_call(0, "TestESDIRK")(out, 0.25, k0)

    np.testing.assert_allclose(out.value, np.array([0.25, 0.5]))
    assert "step" in algebraist.sources["stage1_known_shift_combine"]


def test_implicit_adaptive_empty_combinations_bind_as_no_generated_call() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_implicit_adaptive_scheme(
        known_shifts=(AlgebraistImplicitCombination("stage0_known_shift", (), ()),),
        high_delta=AlgebraistImplicitCombination("high_delta", (), ()),
        low_delta=AlgebraistImplicitCombination("low_delta", (), ()),
        error_delta=AlgebraistImplicitCombination("error_delta", (), ()),
    )

    assert binding.known_shift_calls == (None,)
    assert binding.high_delta_call is None
    assert binding.low_delta_call is None
    assert binding.error_delta_call is None

    with pytest.raises(ValueError, match="no generated algebra"):
        binding.require_known_shift_call(0, "TestESDIRK")
    with pytest.raises(ValueError, match="high-delta"):
        binding.require_high_delta_call("TestESDIRK")
    with pytest.raises(ValueError, match="low-delta"):
        binding.require_low_delta_call("TestESDIRK")
    with pytest.raises(ValueError, match="error-delta"):
        binding.require_error_delta_call("TestESDIRK")

