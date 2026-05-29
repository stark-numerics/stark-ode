import math

import numpy as np

from stark.carriers import CarrierNative, CarrierNumpy
from stark.interface.vector import StarkVector, StarkVectorTranslation, StarkVectorAllocator


def test_stark_vector_creates_translation_with_same_carrier() -> None:
    carrier = CarrierNative([1.0, 2.0])
    vector = StarkVector([1.0, 2.0], carrier)

    translation = vector.translation([3.0, 4.0])

    assert isinstance(translation, StarkVectorTranslation)
    assert translation.value == [3.0, 4.0]
    assert translation.carrier is carrier


def test_stark_vector_zero_translation_uses_carrier_allocation() -> None:
    carrier = CarrierNative([1.0, 2.0])
    vector = StarkVector([1.0, 2.0], carrier)

    translation = vector.zero_translation()

    assert translation.value == [0, 0]
    assert translation.carrier is carrier


def test_stark_vector_allocator_uses_same_carrier() -> None:
    carrier = CarrierNative([1.0, 2.0])
    vector = StarkVector([1.0, 2.0], carrier)

    allocator = vector.allocator()

    assert isinstance(allocator, StarkVectorAllocator)
    assert allocator.carrier is carrier


def test_native_translation_apply_writes_return_result_to_state() -> None:
    carrier = CarrierNative([0.0, 0.0])
    origin = StarkVector([1.0, 2.0], carrier)
    result = StarkVector([0.0, 0.0], carrier)
    translation = StarkVectorTranslation([3.0, 4.0], carrier)

    returned = translation(origin, result)

    assert returned is None
    assert result.value == [4.0, 6.0]


def test_numpy_translation_apply_writes_into_result_state() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    origin = StarkVector(np.array([1.0, 2.0]), carrier)
    result = StarkVector(np.zeros(2), carrier)
    translation = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)

    returned = translation(origin, result)

    assert returned is None
    np.testing.assert_allclose(result.value, [4.0, 6.0])


def test_native_translation_norm_uses_carrier_norm() -> None:
    carrier = CarrierNative([0.0, 0.0])
    translation = StarkVectorTranslation([3.0, 4.0], carrier)

    assert math.isclose(translation.norm(), math.sqrt((9.0 + 16.0) / 2.0))


def test_numpy_translation_norm_uses_carrier_norm() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    translation = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)

    assert math.isclose(translation.norm(), math.sqrt(12.5))


def test_native_translation_addition_returns_new_translation() -> None:
    carrier = CarrierNative([0.0, 0.0])
    left = StarkVectorTranslation([1.0, 2.0], carrier)
    right = StarkVectorTranslation([3.0, 4.0], carrier)

    result = left + right

    assert result.value == [4.0, 6.0]
    assert result.carrier is carrier


def test_numpy_translation_addition_returns_new_translation() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    left = StarkVectorTranslation(np.array([1.0, 2.0]), carrier)
    right = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)

    result = left + right

    np.testing.assert_allclose(result.value, [4.0, 6.0])
    assert result.carrier is carrier


def test_native_translation_scalar_multiply_returns_new_translation() -> None:
    carrier = CarrierNative([0.0, 0.0])
    translation = StarkVectorTranslation([1.0, 2.0], carrier)

    result = 3.0 * translation

    assert result.value == [3.0, 6.0]
    assert result.carrier is carrier


def test_numpy_translation_scalar_multiply_returns_new_translation() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    translation = StarkVectorTranslation(np.array([1.0, 2.0]), carrier)

    result = 3.0 * translation

    np.testing.assert_allclose(result.value, [3.0, 6.0])
    assert result.carrier is carrier


def test_linear_combine_exposes_scale_and_fixed_arity_combiners() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    translation = StarkVectorTranslation(np.array([1.0, 2.0]), carrier)

    linear_combine = translation.linear_combine

    assert len(linear_combine) == 12
    assert linear_combine[0] is translation.scale
    assert linear_combine[1] is translation.combine2
    assert linear_combine[10] is translation.combine11
    assert linear_combine[11] is translation.combine12


def test_numpy_scale_linear_combine_writes_to_output_translation() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    value = StarkVectorTranslation(np.array([1.0, 2.0]), carrier)
    out = StarkVectorTranslation(np.zeros(2), carrier)

    returned = value.scale(3.0, value, out)

    assert returned is out
    np.testing.assert_allclose(out.value, [3.0, 6.0])


def test_numpy_combine2_writes_to_output_translation() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    x0 = StarkVectorTranslation(np.array([1.0, 2.0]), carrier)
    x1 = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)
    out = StarkVectorTranslation(np.zeros(2), carrier)

    returned = x0.combine2(2.0, x0, 3.0, x1, out)

    assert returned is out
    np.testing.assert_allclose(out.value, [11.0, 16.0])


def test_numpy_combine4_writes_to_output_translation() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))
    x0 = StarkVectorTranslation(np.array([1.0, 1.0]), carrier)
    x1 = StarkVectorTranslation(np.array([2.0, 2.0]), carrier)
    x2 = StarkVectorTranslation(np.array([3.0, 3.0]), carrier)
    x3 = StarkVectorTranslation(np.array([4.0, 4.0]), carrier)
    out = StarkVectorTranslation(np.zeros(2), carrier)

    returned = x0.combine4(1.0, x0, 2.0, x1, 3.0, x2, 4.0, x3, out)

    assert returned is out
    np.testing.assert_allclose(out.value, [30.0, 30.0])


def test_allocator_allocates_state() -> None:
    carrier = CarrierNumpy(np.array([1.0, 2.0]))
    allocator = StarkVectorAllocator(carrier)

    state = allocator.allocate_state()

    assert isinstance(state, StarkVector)
    assert state.carrier is carrier
    np.testing.assert_allclose(state.value, [0.0, 0.0])


def test_allocator_allocates_translation() -> None:
    carrier = CarrierNumpy(np.array([1.0, 2.0]))
    allocator = StarkVectorAllocator(carrier)

    translation = allocator.allocate_translation()

    assert isinstance(translation, StarkVectorTranslation)
    assert translation.carrier is carrier
    np.testing.assert_allclose(translation.value, [0.0, 0.0])


def test_allocator_copies_state() -> None:
    carrier = CarrierNumpy(np.array([1.0, 2.0]))
    allocator = StarkVectorAllocator(carrier)
    source = StarkVector(np.array([3.0, 4.0]), carrier)
    result = StarkVector(np.zeros(2), carrier)

    returned = allocator.copy_state(source, result)

    assert returned is result
    np.testing.assert_allclose(result.value, [3.0, 4.0])
    assert result.value is not source.value
