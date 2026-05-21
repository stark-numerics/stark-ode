from __future__ import annotations

import pytest

from stark.carriers import (
    DeprecatedCarrierNative,
    DeprecatedCarrierNativeBound,
    DeprecatedCarrierNumpy,
    DeprecatedCarrierNumpyBound,
    DeprecatedCarrierLibrary,
)


def test_native_accepts_scalar_int() -> None:
    assert DeprecatedCarrierNative().accepts(1)


def test_native_accepts_scalar_float() -> None:
    assert DeprecatedCarrierNative().accepts(1.0)


def test_native_accepts_list_of_numbers() -> None:
    assert DeprecatedCarrierNative().accepts([1, 2.0])


def test_native_accepts_tuple_of_numbers() -> None:
    assert DeprecatedCarrierNative().accepts((1, 2.0))


def test_native_rejects_arbitrary_object() -> None:
    assert not DeprecatedCarrierNative().accepts(object())


def test_native_rejects_mixed_list() -> None:
    assert not DeprecatedCarrierNative().accepts([1.0, object()])


def test_native_coerces_int_scalar_to_float() -> None:
    assert DeprecatedCarrierNative().coerce_state(1) == 1.0


def test_native_coerces_list_values_to_float() -> None:
    assert DeprecatedCarrierNative().coerce_state([1, 2]) == [1.0, 2.0]


def test_native_coerces_tuple_values_to_float() -> None:
    assert DeprecatedCarrierNative().coerce_state((1, 2)) == (1.0, 2.0)


def test_native_bind_scalar() -> None:
    bound = DeprecatedCarrierNative().bind(1)

    assert isinstance(bound, DeprecatedCarrierNativeBound)
    assert bound.template == 1.0


def test_native_bind_list() -> None:
    bound = DeprecatedCarrierNative().bind([1, 2])

    assert isinstance(bound, DeprecatedCarrierNativeBound)
    assert bound.template == [1.0, 2.0]


def test_native_bind_tuple() -> None:
    bound = DeprecatedCarrierNative().bind((1, 2))

    assert isinstance(bound, DeprecatedCarrierNativeBound)
    assert bound.template == (1.0, 2.0)


def test_native_bound_zero_scalar() -> None:
    bound = DeprecatedCarrierNative().bind(1.0)

    assert bound.zero_state() == 0.0
    assert bound.zero_translation() == 0.0


def test_native_bound_zero_list() -> None:
    bound = DeprecatedCarrierNative().bind([1.0, 2.0])

    assert bound.zero_state() == [0.0, 0.0]
    assert bound.zero_translation() == [0.0, 0.0]


def test_native_bound_zero_tuple() -> None:
    bound = DeprecatedCarrierNative().bind((1.0, 2.0))

    assert bound.zero_state() == (0.0, 0.0)
    assert bound.zero_translation() == (0.0, 0.0)


def test_native_bound_copy_list_is_independent() -> None:
    bound = DeprecatedCarrierNative().bind([1.0, 2.0])
    source = [3.0, 4.0]

    copied = bound.copy_state(source)

    assert copied == source
    assert copied is not source


def test_native_bound_copy_tuple() -> None:
    bound = DeprecatedCarrierNative().bind((1.0, 2.0))
    source = (3.0, 4.0)

    assert bound.copy_state(source) == source


def test_native_bound_coerce_translation_scalar() -> None:
    bound = DeprecatedCarrierNative().bind(1.0)

    assert bound.coerce_translation(2) == 2.0


def test_native_bound_coerce_translation_list() -> None:
    bound = DeprecatedCarrierNative().bind([1.0, 2.0])

    assert bound.coerce_translation([3, 4]) == [3.0, 4.0]


def test_native_bound_coerce_translation_tuple() -> None:
    bound = DeprecatedCarrierNative().bind((1.0, 2.0))

    assert bound.coerce_translation((3, 4)) == (3.0, 4.0)


def test_native_bound_validates_wrong_list_length() -> None:
    bound = DeprecatedCarrierNative().bind([1.0, 2.0])

    with pytest.raises(ValueError):
        bound.validate_translation([1.0, 2.0, 3.0])


def test_native_bound_validates_wrong_tuple_length() -> None:
    bound = DeprecatedCarrierNative().bind((1.0, 2.0))

    with pytest.raises(ValueError):
        bound.validate_translation((1.0, 2.0, 3.0))


def test_native_bound_validates_wrong_list_type() -> None:
    bound = DeprecatedCarrierNative().bind([1.0, 2.0])

    with pytest.raises(TypeError):
        bound.validate_translation((1.0, 2.0))


def test_native_bound_validates_wrong_tuple_type() -> None:
    bound = DeprecatedCarrierNative().bind((1.0, 2.0))

    with pytest.raises(TypeError):
        bound.validate_translation([1.0, 2.0])


def test_numpy_accepts_ndarray() -> None:
    np = pytest.importorskip("numpy")

    assert DeprecatedCarrierNumpy().accepts(np.array([1.0, 2.0]))


def test_numpy_does_not_accept_python_scalar() -> None:
    assert not DeprecatedCarrierNumpy().accepts(1.0)


def test_numpy_bind_array() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy().bind(np.array([1.0, 2.0]))

    assert isinstance(bound, DeprecatedCarrierNumpyBound)
    np.testing.assert_allclose(bound.template, np.array([1.0, 2.0]))


def test_numpy_bound_zero_state() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy().bind(np.array([1.0, 2.0]))

    np.testing.assert_allclose(bound.zero_state(), np.array([0.0, 0.0]))
    np.testing.assert_allclose(bound.zero_translation(), np.array([0.0, 0.0]))


def test_numpy_bound_copy_state_is_independent() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy().bind(np.array([1.0, 2.0]))
    source = np.array([3.0, 4.0])

    copied = bound.copy_state(source)

    np.testing.assert_allclose(copied, source)
    assert copied is not source


def test_numpy_bound_coerce_translation() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy().bind(np.array([1.0, 2.0]))
    coerced = bound.coerce_translation([3, 4])

    np.testing.assert_allclose(coerced, np.array([3.0, 4.0]))


def test_numpy_bound_validates_wrong_shape() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy().bind(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        bound.validate_translation(np.array([1.0, 2.0, 3.0]))


def test_numpy_strict_dtype_validates_wrong_dtype() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy(strict_dtype=True).bind(
        np.array([1.0, 2.0], dtype=np.float64)
    )

    with pytest.raises(TypeError):
        bound.validate_translation(np.array([1.0, 2.0], dtype=np.float32))


def test_numpy_dtype_override() -> None:
    np = pytest.importorskip("numpy")

    bound = DeprecatedCarrierNumpy(dtype=np.float32).bind(np.array([1.0, 2.0]))

    assert bound.template.dtype == np.float32

def test_carrier_library_scalar_returns_native() -> None:
    library = DeprecatedCarrierLibrary.default()

    assert isinstance(library.carrier_for(1.0), DeprecatedCarrierNative)


def test_carrier_library_list_returns_native() -> None:
    library = DeprecatedCarrierLibrary.default()

    assert isinstance(library.carrier_for([1.0, 2.0]), DeprecatedCarrierNative)


def test_carrier_library_numpy_array_returns_numpy() -> None:
    np = pytest.importorskip("numpy")

    library = DeprecatedCarrierLibrary.default()

    assert isinstance(library.carrier_for(np.array([1.0, 2.0])), DeprecatedCarrierNumpy)


def test_carrier_library_unknown_object_raises() -> None:
    library = DeprecatedCarrierLibrary.default()

    with pytest.raises(TypeError):
        library.carrier_for(object())


def test_carrier_library_can_be_explicitly_constructed() -> None:
    library = DeprecatedCarrierLibrary((DeprecatedCarrierNative(),))

    assert isinstance(library.carrier_for(1.0), DeprecatedCarrierNative)


def test_carrier_library_respects_order() -> None:
    class CarrierEverything:
        def accepts(self, value: object) -> bool:
            return True

    everything = CarrierEverything()
    library = DeprecatedCarrierLibrary((everything, DeprecatedCarrierNative()))

    assert library.carrier_for(1.0) is everything