from __future__ import annotations

import math

import pytest

from stark.carriers.norms import (
    CarrierNormNativeRMS,
    CarrierNormNumpyMax,
    CarrierNormNumpyRMS,
)


def test_native_scalar_rms() -> None:
    norm = CarrierNormNativeRMS().bind(template=1.0)

    assert norm(-3.0) == 3.0


def test_native_list_rms() -> None:
    norm = CarrierNormNativeRMS().bind(template=[0.0, 0.0])

    assert math.isclose(
        norm([3.0, 4.0]),
        math.sqrt((9.0 + 16.0) / 2.0),
    )


def test_native_tuple_rms() -> None:
    norm = CarrierNormNativeRMS().bind(template=(0.0, 0.0))

    assert math.isclose(
        norm((3.0, 4.0)),
        math.sqrt((9.0 + 16.0) / 2.0),
    )


def test_native_empty_list_rms_is_zero() -> None:
    norm = CarrierNormNativeRMS().bind(template=[])

    assert norm([]) == 0.0


def test_native_empty_tuple_rms_is_zero() -> None:
    norm = CarrierNormNativeRMS().bind(template=())

    assert norm(()) == 0.0


def test_numpy_rms() -> None:
    np = pytest.importorskip("numpy")

    value = np.array([3.0, 4.0])
    norm = CarrierNormNumpyRMS().bind(template=value)

    assert math.isclose(
        norm(value),
        math.sqrt((9.0 + 16.0) / 2.0),
    )


def test_numpy_max() -> None:
    np = pytest.importorskip("numpy")

    value = np.array([-3.0, 4.0])
    norm = CarrierNormNumpyMax().bind(template=value)

    assert norm(value) == 4.0


def test_numpy_empty_rms_is_zero() -> None:
    np = pytest.importorskip("numpy")

    value = np.array([])
    norm = CarrierNormNumpyRMS().bind(template=value)

    assert norm(value) == 0.0


def test_numpy_empty_max_is_zero() -> None:
    np = pytest.importorskip("numpy")

    value = np.array([])
    norm = CarrierNormNumpyMax().bind(template=value)

    assert norm(value) == 0.0


def test_numpy_complex_rms_uses_magnitude() -> None:
    np = pytest.importorskip("numpy")

    value = np.array([3.0 + 4.0j, 0.0 + 0.0j])
    norm = CarrierNormNumpyRMS().bind(template=value)

    assert math.isclose(
        norm(value),
        math.sqrt((25.0 + 0.0) / 2.0),
    )


def test_bound_norm_is_callable() -> None:
    norm = CarrierNormNativeRMS().bind(template=1.0)

    assert callable(norm)