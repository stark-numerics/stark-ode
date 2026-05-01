from __future__ import annotations

import pytest

from stark.carriers import (
    CarrierNative,
    CarrierNativeBound,
    CarrierNumpy,
    CarrierNumpyBound,
    CarrierLibrary,
)


def test_native_accepts_scalar_int() -> None:
    assert CarrierNative().accepts(1)


def test_native_accepts_scalar_float() -> None:
    assert CarrierNative().accepts(1.0)


def test_native_accepts_list_of_numbers() -> None:
    assert CarrierNative().accepts([1, 2.0])


def test_native_accepts_tuple_of_numbers() -> None:
    assert CarrierNative().accepts((1, 2.0))


def test_native_rejects_arbitrary_object() -> None:
    assert not CarrierNative().accepts(object())


def test_native_rejects_mixed_list() -> None:
    assert not CarrierNative().accepts([1.0, object()])


def test_native_coerces_int_scalar_to_float() -> None:
    assert CarrierNative().coerce_state(1) == 1.0


def test_native_coerces_list_values_to_float() -> None:
    assert CarrierNative().coerce_state([1, 2]) == [1.0, 2.0]


def test_native_coerces_tuple_values_to_float() -> None:
    assert CarrierNative().coerce_state((1, 2)) == (1.0, 2.0)


def test_native_bind_scalar() -> None:
    bound = CarrierNative().bind(1)

    assert isinstance(bound, CarrierNativeBound)
    assert bound.template == 1.0


def test_native_bind_list() -> None:
    bound = CarrierNative().bind([1, 2])

    assert isinstance(bound, CarrierNativeBound)
    assert bound.template == [1.0, 2.0]


def test_native_bind_tuple() -> None:
    bound = CarrierNative().bind((1, 2))

    assert isinstance(bound, CarrierNativeBound)
    assert bound.template == (1.0, 2.0)


def test_native_bound_zero_scalar() -> None:
    bound = CarrierNative().bind(1.0)

    assert bound.zero_state() == 0.0
    assert bound.zero_translation() == 0.0


def test_native_bound_zero_list() -> None:
    bound = CarrierNative().bind([1.0, 2.0])

    assert bound.zero_state() == [0.0, 0.0]
    assert bound.zero_translation() == [0.0, 0.0]


def test_native_bound_zero_tuple() -> None:
    bound = CarrierNative().bind((1.0, 2.0))

    assert bound.zero_state() == (0.0, 0.0)
    assert bound.zero_translation() == (0.0, 0.0)


def test_native_bound_copy_list_is_independent() -> None:
    bound = CarrierNative().bind([1.0, 2.0])
    source = [3.0, 4.0]

    copied = bound.copy_state(source)

    assert copied == source
    assert copied is not source


def test_native_bound_copy_tuple() -> None:
    bound = CarrierNative().bind((1.0, 2.0))
    source = (3.0, 4.0)

    assert bound.copy_state(source) == source


def test_native_bound_coerce_translation_scalar() -> None:
    bound = CarrierNative().bind(1.0)

    assert bound.coerce_translation(2) == 2.0


def test_native_bound_coerce_translation_list() -> None:
    bound = CarrierNative().bind([1.0, 2.0])

    assert bound.coerce_translation([3, 4]) == [3.0, 4.0]


def test_native_bound_coerce_translation_tuple() -> None:
    bound = CarrierNative().bind((1.0, 2.0))

    assert bound.coerce_translation((3, 4)) == (3.0, 4.0)


def test_native_bound_validates_wrong_list_length() -> None:
    bound = CarrierNative().bind([1.0, 2.0])

    with pytest.raises(ValueError):
        bound.validate_translation([1.0, 2.0, 3.0])


def test_native_bound_validates_wrong_tuple_length() -> None:
    bound = CarrierNative().bind((1.0, 2.0))

    with pytest.raises(ValueError):
        bound.validate_translation((1.0, 2.0, 3.0))


def test_native_bound_validates_wrong_list_type() -> None:
    bound = CarrierNative().bind([1.0, 2.0])

    with pytest.raises(TypeError):
        bound.validate_translation((1.0, 2.0))


def test_native_bound_validates_wrong_tuple_type() -> None:
    bound = CarrierNative().bind((1.0, 2.0))

    with pytest.raises(TypeError):
        bound.validate_translation([1.0, 2.0])


def test_numpy_accepts_ndarray() -> None:
    np = pytest.importorskip("numpy")

    assert CarrierNumpy().accepts(np.array([1.0, 2.0]))


def test_numpy_does_not_accept_python_scalar() -> None:
    assert not CarrierNumpy().accepts(1.0)


def test_numpy_bind_array() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy().bind(np.array([1.0, 2.0]))

    assert isinstance(bound, CarrierNumpyBound)
    np.testing.assert_allclose(bound.template, np.array([1.0, 2.0]))


def test_numpy_bound_zero_state() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy().bind(np.array([1.0, 2.0]))

    np.testing.assert_allclose(bound.zero_state(), np.array([0.0, 0.0]))
    np.testing.assert_allclose(bound.zero_translation(), np.array([0.0, 0.0]))


def test_numpy_bound_copy_state_is_independent() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy().bind(np.array([1.0, 2.0]))
    source = np.array([3.0, 4.0])

    copied = bound.copy_state(source)

    np.testing.assert_allclose(copied, source)
    assert copied is not source


def test_numpy_bound_coerce_translation() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy().bind(np.array([1.0, 2.0]))
    coerced = bound.coerce_translation([3, 4])

    np.testing.assert_allclose(coerced, np.array([3.0, 4.0]))


def test_numpy_bound_validates_wrong_shape() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy().bind(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        bound.validate_translation(np.array([1.0, 2.0, 3.0]))


def test_numpy_strict_dtype_validates_wrong_dtype() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy(strict_dtype=True).bind(
        np.array([1.0, 2.0], dtype=np.float64)
    )

    with pytest.raises(TypeError):
        bound.validate_translation(np.array([1.0, 2.0], dtype=np.float32))


def test_numpy_dtype_override() -> None:
    np = pytest.importorskip("numpy")

    bound = CarrierNumpy(dtype=np.float32).bind(np.array([1.0, 2.0]))

    assert bound.template.dtype == np.float32

def test_carrier_library_scalar_returns_native() -> None:
    library = CarrierLibrary.default()

    assert isinstance(library.carrier_for(1.0), CarrierNative)


def test_carrier_library_list_returns_native() -> None:
    library = CarrierLibrary.default()

    assert isinstance(library.carrier_for([1.0, 2.0]), CarrierNative)


def test_carrier_library_numpy_array_returns_numpy() -> None:
    np = pytest.importorskip("numpy")

    library = CarrierLibrary.default()

    assert isinstance(library.carrier_for(np.array([1.0, 2.0])), CarrierNumpy)


def test_carrier_library_unknown_object_raises() -> None:
    library = CarrierLibrary.default()

    with pytest.raises(TypeError):
        library.carrier_for(object())


def test_carrier_library_can_be_explicitly_constructed() -> None:
    library = CarrierLibrary((CarrierNative(),))

    assert isinstance(library.carrier_for(1.0), CarrierNative)


def test_carrier_library_respects_order() -> None:
    class CarrierEverything:
        def accepts(self, value: object) -> bool:
            return True

    everything = CarrierEverything()
    library = CarrierLibrary((everything, CarrierNative()))

    assert library.carrier_for(1.0) is everything