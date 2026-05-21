from __future__ import annotations

import math

import pytest

from stark.carriers.deprecated.kernels import DeprecatedCarrierKernelNative, DeprecatedCarrierKernelNumpy
from stark.carriers.deprecated.norms import DeprecatedCarrierNormNativeRMS, DeprecatedCarrierNormNumpyRMS


def test_native_scalar_translate() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=0.0)
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.translate(2.0, 3.0) == 5.0


def test_native_scalar_add() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=0.0)
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.add(2.0, 3.0) == 5.0


def test_native_scalar_scale() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=0.0)
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.scale(2.0, 3.0) == 6.0


def test_native_scalar_combine() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=0.0)
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.combine((2.0, -1.0), (3.0, 4.0)) == 2.0


def test_native_list_translate() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=[0.0, 0.0])
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.translate([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]


def test_native_list_add() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=[0.0, 0.0])
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]


def test_native_list_scale() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=[0.0, 0.0])
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.scale(2.0, [3.0, 4.0]) == [6.0, 8.0]


def test_native_list_combine() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=[0.0, 0.0])
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.combine(
        (2.0, -1.0),
        ([3.0, 4.0], [1.0, 2.0]),
    ) == [5.0, 6.0]


def test_native_tuple_translate() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=(0.0, 0.0))
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.translate((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)


def test_native_tuple_add() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=(0.0, 0.0))
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.add((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)


def test_native_tuple_scale() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=(0.0, 0.0))
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.scale(2.0, (3.0, 4.0)) == (6.0, 8.0)


def test_native_tuple_combine() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=(0.0, 0.0))
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert kernel.combine(
        (2.0, -1.0),
        ((3.0, 4.0), (1.0, 2.0)),
    ) == (5.0, 6.0)


def test_native_combine_empty_values_raises() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=0.0)
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    with pytest.raises(ValueError):
        kernel.combine((), ())


def test_native_norm_delegates_to_bound_norm() -> None:
    norm = DeprecatedCarrierNormNativeRMS().bind(template=[0.0, 0.0])
    kernel = DeprecatedCarrierKernelNative().bind(norm=norm)

    assert math.isclose(
        kernel.norm([3.0, 4.0]),
        math.sqrt((9.0 + 16.0) / 2.0),
    )


def test_numpy_translate() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    np.testing.assert_allclose(
        kernel.translate(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        np.array([4.0, 6.0]),
    )


def test_numpy_add() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    np.testing.assert_allclose(
        kernel.add(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        np.array([4.0, 6.0]),
    )


def test_numpy_scale() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    np.testing.assert_allclose(
        kernel.scale(2.0, np.array([3.0, 4.0])),
        np.array([6.0, 8.0]),
    )


def test_numpy_combine() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    np.testing.assert_allclose(
        kernel.combine(
            (2.0, -1.0),
            (
                np.array([3.0, 4.0]),
                np.array([1.0, 2.0]),
            ),
        ),
        np.array([5.0, 6.0]),
    )


def test_numpy_combine_empty_values_raises() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    with pytest.raises(ValueError):
        kernel.combine((), ())


def test_numpy_translate_into() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    result = np.zeros(2)
    kernel.translate_into(
        result,
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )

    np.testing.assert_allclose(result, np.array([4.0, 6.0]))


def test_numpy_add_into() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    result = np.zeros(2)
    kernel.add_into(
        result,
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )

    np.testing.assert_allclose(result, np.array([4.0, 6.0]))


def test_numpy_scale_into() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    result = np.zeros(2)
    kernel.scale_into(result, 2.0, np.array([3.0, 4.0]))

    np.testing.assert_allclose(result, np.array([6.0, 8.0]))


def test_numpy_combine_into() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    result = np.zeros(2)
    kernel.combine_into(
        result,
        (2.0, -1.0),
        (
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0]),
        ),
    )

    np.testing.assert_allclose(result, np.array([5.0, 6.0]))


def test_numpy_combine_into_empty_values_raises() -> None:
    np = pytest.importorskip("numpy")

    template = np.array([0.0, 0.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=template)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    with pytest.raises(ValueError):
        kernel.combine_into(np.zeros(2), (), ())


def test_numpy_norm_delegates_to_bound_norm() -> None:
    np = pytest.importorskip("numpy")

    value = np.array([3.0, 4.0])
    norm = DeprecatedCarrierNormNumpyRMS().bind(template=value)
    kernel = DeprecatedCarrierKernelNumpy().bind(norm=norm)

    assert math.isclose(
        kernel.norm(value),
        math.sqrt((9.0 + 16.0) / 2.0),
    )