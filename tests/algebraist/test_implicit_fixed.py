import numpy as np
import pytest

from stark.algebraist import (
    Algebraist,
    AlgebraistField,
    AlgebraistImplicitCombination,
    AlgebraistImplicitFixedSchemeBinding,
)


class FakeTranslation:
    def __init__(self, value):
        self.value = value


def make_algebraist() -> Algebraist:
    return Algebraist(fields=(AlgebraistField("value", "value"),))


def test_implicit_fixed_binding_generates_known_final_and_error_calls() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_implicit_fixed_scheme(
        known_shifts=(
            None,
            AlgebraistImplicitCombination("stage1_known_shift", (0.5,), (0,)),
            AlgebraistImplicitCombination("stage2_known_shift", (-1.0, 2.0), (0, 1)),
        ),
        final_delta=AlgebraistImplicitCombination("final_delta", (0.25, 0.75), (0, 2)),
        error_delta=AlgebraistImplicitCombination("error_delta", (0.25, -0.25), (1, 2)),
    )

    assert isinstance(binding, AlgebraistImplicitFixedSchemeBinding)
    assert binding.known_shift_calls[0] is None

    out = FakeTranslation(np.zeros(2))
    k0 = FakeTranslation(np.array([2.0, 4.0]))
    k1 = FakeTranslation(np.array([6.0, 8.0]))

    binding.require_known_shift_call(1, "TestDIRK")(k0, out)
    np.testing.assert_allclose(out.value, np.array([1.0, 2.0]))

    binding.require_known_shift_call(2, "TestDIRK")(k0, k1, out)
    np.testing.assert_allclose(out.value, np.array([10.0, 12.0]))

    binding.require_final_delta_call("TestDIRK")(k0, k1, out)
    np.testing.assert_allclose(out.value, np.array([5.0, 7.0]))

    binding.require_error_delta_call("TestDIRK")(k0, k1, out)
    np.testing.assert_allclose(out.value, np.array([-1.0, -1.0]))

    assert "stage1_known_shift_combine" in algebraist.sources
    assert "stage1_known_shift_combine_kernel" in algebraist.sources
    assert "final_delta_combine" in algebraist.sources
    assert "error_delta_combine" in algebraist.sources


def test_implicit_fixed_binding_generates_step_scaled_call() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_implicit_fixed_scheme(
        known_shifts=(
            AlgebraistImplicitCombination(
                "known_rhs",
                (0.5,),
                step_scale=True,
            ),
        ),
    )
    out = FakeTranslation(np.zeros(2))
    k0 = FakeTranslation(np.array([2.0, 4.0]))

    binding.require_known_shift_call(0, "TestCN")(0.25, k0, out)

    np.testing.assert_allclose(out.value, np.array([0.25, 0.5]))
    assert "known_rhs_combine" in algebraist.sources
    assert "step" in algebraist.sources["known_rhs_combine"]


def test_implicit_fixed_combination_can_drop_zero_coefficients() -> None:
    combination = AlgebraistImplicitCombination.from_coefficients(
        "stage_known_shift",
        (0.0, 0.5, 0.0, -1.0),
    )

    assert combination.coefficients == (0.5, -1.0)
    assert combination.term_indices == (1, 3)


def test_implicit_fixed_empty_combinations_bind_as_no_generated_call() -> None:
    algebraist = make_algebraist()

    binding = algebraist.bind_implicit_fixed_scheme(
        known_shifts=(AlgebraistImplicitCombination("stage0_known_shift", (), ()),),
        final_delta=AlgebraistImplicitCombination("final_delta", (), ()),
        error_delta=AlgebraistImplicitCombination("error_delta", (), ()),
    )

    assert binding.known_shift_calls == (None,)
    assert binding.final_delta_call is None
    assert binding.error_delta_call is None

    with pytest.raises(ValueError, match="no generated algebra"):
        binding.require_known_shift_call(0, "TestDIRK")
    with pytest.raises(ValueError, match="final-delta"):
        binding.require_final_delta_call("TestDIRK")
    with pytest.raises(ValueError, match="error-delta"):
        binding.require_error_delta_call("TestDIRK")


def test_implicit_fixed_combination_rejects_invalid_descriptors() -> None:
    with pytest.raises(ValueError, match="role"):
        AlgebraistImplicitCombination("stage-1", (1.0,), (0,))

    with pytest.raises(ValueError, match="matching lengths"):
        AlgebraistImplicitCombination("stage1", (1.0,), (0, 1))

    with pytest.raises(ValueError, match="non-negative"):
        AlgebraistImplicitCombination("stage1", (1.0,), (-1,))
