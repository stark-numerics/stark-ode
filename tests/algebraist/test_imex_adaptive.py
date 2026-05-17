import numpy as np
import pytest

from stark.algebraist import (
    Algebraist,
    AlgebraistField,
    AlgebraistImExAdaptiveSchemeBinding,
    AlgebraistImExCombination,
)


class FakeTranslation:
    def __init__(self, value):
        self.value = value


def make_algebraist() -> Algebraist:
    return Algebraist(fields=(AlgebraistField("value", "value"),))


def test_imex_adaptive_binding_generates_split_stage_and_delta_calls() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_imex_adaptive_scheme(
        stage_shifts=(
            None,
            AlgebraistImExCombination.from_coefficients(
                "stage1_shift",
                explicit=(0.5, 0.0),
                implicit=(0.0, -0.25),
            ),
        ),
        high_delta=AlgebraistImExCombination.from_coefficients(
            "high_delta",
            explicit=(0.25, 0.0),
            implicit=(0.0, 0.75),
        ),
        low_delta=AlgebraistImExCombination.from_coefficients(
            "low_delta",
            explicit=(0.5, 0.0),
            implicit=(0.0, 0.5),
        ),
        error_delta=AlgebraistImExCombination.from_coefficients(
            "error_delta",
            explicit=(-0.25, 0.0),
            implicit=(0.0, 0.25),
        ),
    )

    assert isinstance(binding, AlgebraistImExAdaptiveSchemeBinding)
    assert binding.stage_shift_calls[0] is None

    out = FakeTranslation(np.zeros(2))
    explicit_k0 = FakeTranslation(np.array([2.0, 4.0]))
    implicit_k1 = FakeTranslation(np.array([6.0, 8.0]))

    binding.require_stage_shift_call(1, "TestIMEX")(
        0.5,
        explicit_k0,
        implicit_k1,
        out,
    )
    np.testing.assert_allclose(out.value, np.array([-0.25, 0.0]))

    binding.require_high_delta_call("TestIMEX")(
        0.5,
        explicit_k0,
        implicit_k1,
        out,
    )
    np.testing.assert_allclose(out.value, np.array([2.5, 3.5]))

    binding.require_low_delta_call("TestIMEX")(
        0.5,
        explicit_k0,
        implicit_k1,
        out,
    )
    np.testing.assert_allclose(out.value, np.array([2.0, 3.0]))

    binding.require_error_delta_call("TestIMEX")(
        0.5,
        explicit_k0,
        implicit_k1,
        out,
    )
    np.testing.assert_allclose(out.value, np.array([0.5, 0.5]))

    assert "stage1_shift_combine" in algebraist.sources
    assert "high_delta_combine" in algebraist.sources
    assert "low_delta_combine" in algebraist.sources
    assert "error_delta_combine" in algebraist.sources
    assert "explicit_k0" in algebraist.sources["stage1_shift_combine"]
    assert "implicit_k1" in algebraist.sources["stage1_shift_combine"]
    assert "step" in algebraist.sources["stage1_shift_combine"]


def test_imex_adaptive_empty_combinations_bind_as_no_generated_call() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_imex_adaptive_scheme(
        stage_shifts=(
            AlgebraistImExCombination("stage0_shift", (), ()),
        ),
        high_delta=AlgebraistImExCombination("high_delta", (), ()),
        low_delta=AlgebraistImExCombination("low_delta", (), ()),
        error_delta=AlgebraistImExCombination("error_delta", (), ()),
    )

    assert binding.stage_shift_calls == (None,)
    assert binding.high_delta_call is None
    assert binding.low_delta_call is None
    assert binding.error_delta_call is None

    with pytest.raises(ValueError, match="no generated algebra"):
        binding.require_stage_shift_call(0, "TestIMEX")
    with pytest.raises(ValueError, match="high-delta"):
        binding.require_high_delta_call("TestIMEX")
    with pytest.raises(ValueError, match="low-delta"):
        binding.require_low_delta_call("TestIMEX")
    with pytest.raises(ValueError, match="error-delta"):
        binding.require_error_delta_call("TestIMEX")


def test_imex_adaptive_combination_rejects_invalid_descriptors() -> None:
    with pytest.raises(ValueError, match="role"):
        AlgebraistImExCombination("stage-1", (1.0,), ())

    with pytest.raises(ValueError, match="explicit"):
        AlgebraistImExCombination("stage1", (1.0,), (), (0, 1))

    with pytest.raises(ValueError, match="implicit"):
        AlgebraistImExCombination("stage1", (), (1.0,), None, (0, 1))

    with pytest.raises(ValueError, match="non-negative"):
        AlgebraistImExCombination("stage1", (1.0,), (), (-1,), ())
