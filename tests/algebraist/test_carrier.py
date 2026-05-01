from dataclasses import dataclass

import numpy as np
import pytest

from stark.carriers.algebraist import (
    CarrierKernelAlgebraist,
    CarrierKernelAlgebraistBound,
)
from stark.algebraist import AlgebraistField


VALUE_FIELD = AlgebraistField("value", "value")


@dataclass
class ArrayBox:
    value: np.ndarray


class DummyNorm:
    def __call__(self, value):
        return float(np.sqrt(np.mean(value.value**2)))


def test_algebraist_kernel_binds():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")

    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    assert isinstance(bound, CarrierKernelAlgebraistBound)


def test_algebraist_kernel_translate_into():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    origin = ArrayBox(np.array([1.0, 2.0]))
    delta = ArrayBox(np.array([3.0, 4.0]))
    result = ArrayBox(np.zeros(2))

    bound.translate_into(result, origin, delta)

    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))


def test_algebraist_kernel_add_into():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    left = ArrayBox(np.array([1.0, 2.0]))
    right = ArrayBox(np.array([3.0, 4.0]))
    result = ArrayBox(np.zeros(2))

    bound.add_into(result, left, right)

    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))


def test_algebraist_kernel_scale_into():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    value = ArrayBox(np.array([3.0, 4.0]))
    result = ArrayBox(np.zeros(2))

    bound.scale_into(result, 2.0, value)

    np.testing.assert_allclose(result.value, np.array([6.0, 8.0]))


def test_algebraist_kernel_combine_into():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    left = ArrayBox(np.array([1.0, 2.0]))
    right = ArrayBox(np.array([3.0, 4.0]))
    result = ArrayBox(np.zeros(2))

    bound.combine_into(result, [2.0, 3.0], [left, right])

    np.testing.assert_allclose(result.value, np.array([11.0, 16.0]))


def test_algebraist_kernel_empty_combine_into_raises():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    with pytest.raises(ValueError):
        bound.combine_into(ArrayBox(np.zeros(2)), [], [])


def test_algebraist_kernel_combine_into_too_many_values_raises():
    kernel = CarrierKernelAlgebraist(
        fields=[VALUE_FIELD],
        fused_up_to=2,
        generate_norm="rms",
    )
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    values = [
        ArrayBox(np.array([1.0, 2.0])),
        ArrayBox(np.array([3.0, 4.0])),
        ArrayBox(np.array([5.0, 6.0])),
    ]

    with pytest.raises(ValueError):
        bound.combine_into(ArrayBox(np.zeros(2)), [1.0, 1.0, 1.0], values)


def test_algebraist_kernel_norm_uses_generated_norm_when_available():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    result = bound.norm(ArrayBox(np.array([3.0, 4.0])))

    assert result == pytest.approx(5.0)

def test_algebraist_kernel_norm_falls_back_to_bound_norm_policy():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm=None)
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    result = bound.norm(ArrayBox(np.array([3.0, 4.0])))

    assert result == pytest.approx((12.5) ** 0.5)


def test_algebraist_kernel_returning_methods_raise():
    kernel = CarrierKernelAlgebraist(fields=[VALUE_FIELD], generate_norm="rms")
    bound = kernel.bind(
        template=ArrayBox(np.array([1.0, 2.0])),
        carrier=None,
        norm_policy=DummyNorm(),
    )

    with pytest.raises(NotImplementedError):
        bound.translate(ArrayBox(np.zeros(2)), ArrayBox(np.zeros(2)))

    with pytest.raises(NotImplementedError):
        bound.add(ArrayBox(np.zeros(2)), ArrayBox(np.zeros(2)))

    with pytest.raises(NotImplementedError):
        bound.scale(2.0, ArrayBox(np.zeros(2)))

    with pytest.raises(NotImplementedError):
        bound.combine([1.0], [ArrayBox(np.zeros(2))])


def test_carrier_kernel_algebraist_requires_algebraist_or_fields():
    with pytest.raises(ValueError):
        CarrierKernelAlgebraist()
