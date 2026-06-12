from __future__ import annotations

from array import array

import numpy as np
import pytest

from stark.engines.carriers import CarrierNative, CarrierNumpy
from stark.engines.carriers.native import (
    CarrierBasisNativeArray,
    CarrierBasisNativeList,
    CarrierBasisNativeScalar,
    CarrierBasisNativeTuple,
    CarrierNativeArray,
    CarrierNativeList,
    CarrierNativeScalar,
    CarrierNativeTuple,
)
from stark.core.contracts import TranslationBasis


def assert_dual_coordinates(basis: TranslationBasis, output, values) -> None:
    assert basis.dimension == len(values)
    for column, expected in enumerate(values):
        vector = basis.vector(column, output)
        assert vector == expected
        for row in range(basis.dimension):
            assert basis.coordinate(row, vector) == pytest.approx(1.0 if row == column else 0.0)


def test_native_scalar_basis() -> None:
    carrier = CarrierNativeScalar(0.0)

    assert isinstance(carrier.basis, CarrierBasisNativeScalar)
    assert carrier.basis.dimension == 1
    assert carrier.basis.vector(0, 0.0) == 1.0
    assert carrier.basis.coordinate(0, 7.0) == 7.0

    with pytest.raises(IndexError):
        carrier.basis.vector(1, 0.0)


def test_native_list_basis() -> None:
    carrier = CarrierNativeList([0.0, 0.0, 0.0])

    assert isinstance(carrier.basis, CarrierBasisNativeList)
    assert_dual_coordinates(
        carrier.basis,
        carrier.allocation.zero_translation(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )


def test_native_tuple_basis() -> None:
    carrier = CarrierNativeTuple((0.0, 0.0, 0.0))

    assert isinstance(carrier.basis, CarrierBasisNativeTuple)
    assert_dual_coordinates(
        carrier.basis,
        carrier.allocation.zero_translation(),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def test_native_array_basis() -> None:
    carrier = CarrierNativeArray(array("d", [0.0, 0.0, 0.0]))
    output = carrier.allocation.zero_translation()

    assert isinstance(carrier.basis, CarrierBasisNativeArray)
    assert carrier.basis.dimension == 3

    vector = carrier.basis.vector(2, output)

    assert vector is output
    assert list(vector) == [0.0, 0.0, 1.0]
    assert carrier.basis.coordinate(2, vector) == pytest.approx(1.0)


def test_native_facade_basis_delegates_to_concrete_basis() -> None:
    carrier = CarrierNative([0.0, 0.0])

    assert carrier.basis.dimension == 2
    assert carrier.basis.vector(1, carrier.allocation.zero_translation()) == [0.0, 1.0]


def test_numpy_basis_uses_canonical_flat_coordinates() -> None:
    carrier = CarrierNumpy(np.zeros((2, 2)))
    output = carrier.allocation.zero_translation()

    assert carrier.basis.dimension == 4

    vector = carrier.basis.vector(2, output)

    assert vector is output
    np.testing.assert_allclose(vector, [[0.0, 0.0], [1.0, 0.0]])
    assert carrier.basis.coordinate(2, vector) == pytest.approx(1.0)
    assert carrier.basis.coordinate(1, vector) == pytest.approx(0.0)


def test_cupy_basis_optional() -> None:
    cp = pytest.importorskip("cupy")
    from stark.engines.carriers import CarrierCupy

    carrier = CarrierCupy(cp.zeros((2, 2)))
    output = carrier.allocation.zero_translation()

    vector = carrier.basis.vector(3, output)

    assert vector is output
    assert carrier.basis.dimension == 4
    np.testing.assert_allclose(cp.asnumpy(vector), [[0.0, 0.0], [0.0, 1.0]])
    assert carrier.basis.coordinate(3, vector) == pytest.approx(1.0)


def test_jax_basis_optional() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from stark.engines.carriers import CarrierJax

    carrier = CarrierJax(jnp.zeros((2, 2)))
    output = carrier.allocation.zero_translation()

    vector = carrier.basis.vector(3, output)

    assert vector is not output
    assert carrier.basis.dimension == 4
    np.testing.assert_allclose(np.asarray(vector), [[0.0, 0.0], [0.0, 1.0]])
    assert carrier.basis.coordinate(3, vector) == pytest.approx(1.0)


def test_numpy_basis_analyses_and_synthesizes_coordinates() -> None:
    carrier = CarrierNumpy(np.zeros((2, 2)))
    coordinates = [0.0, 0.0, 0.0, 0.0]

    carrier.basis.coordinates(np.array([[1.0, 2.0], [3.0, 4.0]]), coordinates)

    assert coordinates == pytest.approx([1.0, 2.0, 3.0, 4.0])

    output = carrier.allocation.zero_translation()
    synthesized = carrier.basis.synthesize([5.0, 6.0, 7.0, 8.0], output)

    assert synthesized is output
    np.testing.assert_allclose(synthesized, [[5.0, 6.0], [7.0, 8.0]])


def test_jax_basis_synthesizes_return_style_translation() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from stark.engines.carriers import CarrierJax

    carrier = CarrierJax(jnp.zeros((2, 2)))
    output = carrier.allocation.zero_translation()

    synthesized = carrier.basis.synthesize([1.0, 2.0, 3.0, 4.0], output)

    assert synthesized is not output
    np.testing.assert_allclose(np.asarray(synthesized), [[1.0, 2.0], [3.0, 4.0]])
