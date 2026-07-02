import math

import numpy as np
import pytest

from stark.engines.native.carriers import CarrierNative

from stark.engines.numpy.carriers import CarrierNumpy
def test_native_scalar_carrier_parts() -> None:
    carrier = CarrierNative(2.0)

    assert carrier.storage.is_state(1.0)
    assert carrier.storage.is_translation(3.0)

    assert carrier.validation.validate_state(2.0) == 2.0
    assert carrier.validation.coerce_translation(4.0) == 4.0

    assert carrier.allocation.zero_state() == 0
    assert carrier.allocation.zero_translation() == 0
    assert carrier.allocation.copy_state(5.0) == 5.0

    result = 0.0
    assert carrier.arithmetic.preference == "return"
    assert carrier.arithmetic.translate(2.0, 0.5, 4.0, result) == 4.0
    assert carrier.arithmetic.add(2.0, 3.0, result) == 5.0
    assert carrier.arithmetic.scale(3.0, 2.0, result) == 6.0
    assert carrier.arithmetic.combine3(1.0, 2.0, 2.0, 3.0, 3.0, 4.0, result) == 20.0

    assert carrier.norm(-3.0) == 3.0


def test_native_sequence_carrier_parts() -> None:
    carrier = CarrierNative([1.0, 2.0, 3.0])

    assert carrier.storage.is_state([0.0, 0.0, 0.0])
    assert not carrier.storage.is_state([0.0, 0.0])

    assert carrier.validation.validate_translation([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    assert carrier.allocation.zero_translation() == [0, 0, 0]
    assert carrier.allocation.copy_translation([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    result = [0.0, 0.0, 0.0]
    assert carrier.arithmetic.translate([1.0, 2.0, 3.0], 2.0, [4.0, 5.0, 6.0], result) == [9.0, 12.0, 15.0]
    assert carrier.arithmetic.add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], result) == [5.0, 7.0, 9.0]
    assert carrier.arithmetic.scale(2.0, [1.0, 2.0, 3.0], result) == [2.0, 4.0, 6.0]
    assert carrier.arithmetic.combine2(2.0, [1.0, 2.0, 3.0], 3.0, [4.0, 5.0, 6.0], result) == [14.0, 19.0, 24.0]

    assert math.isclose(carrier.norm([3.0, 4.0]), math.sqrt((9.0 + 16.0) / 2.0))


def test_numpy_carrier_parts() -> None:
    template = np.array([1.0, 2.0, 3.0])
    carrier = CarrierNumpy(template)

    assert carrier.storage.is_state(np.array([0.0, 0.0, 0.0]))
    assert not carrier.storage.is_state(np.array([0.0, 0.0]))

    state = np.array([1.0, 2.0, 3.0])
    translation = np.array([4.0, 5.0, 6.0])

    assert carrier.validation.validate_state(state) is state
    np.testing.assert_allclose(
        carrier.validation.coerce_translation([4.0, 5.0, 6.0]),
        translation,
    )

    np.testing.assert_allclose(carrier.allocation.zero_state(), np.zeros(3))
    copied = carrier.allocation.copy_translation(translation)
    assert copied is not translation
    np.testing.assert_allclose(copied, translation)

    assert carrier.arithmetic.preference == "into"


def test_numpy_arithmetic_writes_into_result() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0, 0.0]))

    state = np.array([1.0, 2.0, 3.0])
    derivative = np.array([4.0, 5.0, 6.0])
    result = np.zeros(3)

    returned = carrier.arithmetic.translate(state, 2.0, derivative, result)

    assert returned is None
    np.testing.assert_allclose(result, [9.0, 12.0, 15.0])

    carrier.arithmetic.add(state, derivative, result)
    np.testing.assert_allclose(result, [5.0, 7.0, 9.0])

    carrier.arithmetic.scale(3.0, state, result)
    np.testing.assert_allclose(result, [3.0, 6.0, 9.0])

    carrier.arithmetic.combine4(
        1.0, state,
        2.0, derivative,
        3.0, np.array([1.0, 1.0, 1.0]),
        4.0, np.array([2.0, 2.0, 2.0]),
        result,
    )
    np.testing.assert_allclose(result, [20.0, 23.0, 26.0])


def test_numpy_translate_allows_state_result_aliasing() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0, 0.0]))

    state = np.array([1.0, 2.0, 3.0])
    derivative = np.array([4.0, 5.0, 6.0])

    returned = carrier.arithmetic.translate(state, 2.0, derivative, state)

    assert returned is None
    np.testing.assert_allclose(state, [9.0, 12.0, 15.0])


def test_numpy_norms_rms_default() -> None:
    carrier = CarrierNumpy(np.array([0.0, 0.0]))

    assert math.isclose(carrier.norm(np.array([3.0, 4.0])), math.sqrt(12.5))


def test_cupy_carrier_optional() -> None:
    cp = pytest.importorskip("cupy")
    from stark.engines.cupy.carriers import CarrierCupy
    carrier = CarrierCupy(cp.asarray([1.0, 2.0, 3.0]))

    result = carrier.allocation.zero_translation()
    carrier.arithmetic.combine2(
        2.0, cp.asarray([1.0, 2.0, 3.0]),
        3.0, cp.asarray([4.0, 5.0, 6.0]),
        result,
    )

    np.testing.assert_allclose(cp.asnumpy(result), [14.0, 19.0, 24.0])


def test_cupy_translate_allows_state_result_aliasing_optional() -> None:
    cp = pytest.importorskip("cupy")
    from stark.engines.cupy.carriers import CarrierCupy
    carrier = CarrierCupy(cp.asarray([1.0, 2.0, 3.0]))

    state = cp.asarray([1.0, 2.0, 3.0])
    derivative = cp.asarray([4.0, 5.0, 6.0])
    returned = carrier.arithmetic.translate(state, 2.0, derivative, state)

    assert returned is None
    np.testing.assert_allclose(cp.asnumpy(state), [9.0, 12.0, 15.0])


def test_jax_carrier_optional() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from stark.engines.jax.carriers import CarrierJax
    carrier = CarrierJax(jnp.asarray([1.0, 2.0, 3.0]))

    result = carrier.allocation.zero_translation()
    returned = carrier.arithmetic.combine2(
        2.0, jnp.asarray([1.0, 2.0, 3.0]),
        3.0, jnp.asarray([4.0, 5.0, 6.0]),
        result,
    )

    assert returned is not None
    np.testing.assert_allclose(np.asarray(returned), [14.0, 19.0, 24.0])
