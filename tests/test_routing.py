import numpy as np
import pytest

from stark.carriers import CarrierNative, CarrierNumpy
from stark.routing import (
    Routing,
    RoutingVector,
    RoutingVectorInPlace,
    RoutingVectorPreferInPlace,
    RoutingVectorReturn,
)


class Box:
    def __init__(self, value):
        self.value = value


class ReturningKernel:
    def translate(self, origin, delta):
        return origin + delta

    def add(self, left, right):
        return left + right

    def scale(self, scalar, value):
        return scalar * value

    def combine(self, coefficients, values):
        if not values:
            raise ValueError("Cannot combine empty values.")

        result = coefficients[0] * values[0]
        for coefficient, value in zip(coefficients[1:], values[1:]):
            result = result + coefficient * value
        return result


class InPlaceKernel(ReturningKernel):
    def translate_into(self, result, origin, delta):
        result[...] = origin + delta

    def add_into(self, result, left, right):
        result[...] = left + right

    def scale_into(self, result, scalar, value):
        result[...] = scalar * value

    def combine_into(self, result, coefficients, values):
        if not values:
            raise ValueError("Cannot combine empty values.")

        result[...] = coefficients[0] * values[0]
        for coefficient, value in zip(coefficients[1:], values[1:]):
            result[...] = result + coefficient * value


def test_routing_default_has_vector_return_policy():
    routing = Routing.default()

    assert isinstance(routing.policy(RoutingVector), RoutingVectorReturn)


def test_routing_can_override_vector_policy():
    routing = Routing(vector=RoutingVectorPreferInPlace())

    assert isinstance(routing.policy(RoutingVector), RoutingVectorPreferInPlace)


def test_routing_unknown_policy_family_raises():
    routing = Routing.default()

    with pytest.raises(KeyError):
        routing.policy(object)


def test_vector_return_uses_returning_translate():
    routing = RoutingVectorReturn()
    kernel = ReturningKernel()

    result = Box(0.0)
    origin = Box(2.0)
    delta = Box(3.0)

    routing.translate(kernel, result, origin, delta)

    assert result.value == 5.0


def test_vector_return_uses_returning_add_scale_and_combine():
    routing = RoutingVectorReturn()
    kernel = ReturningKernel()

    result = Box(0.0)

    routing.add(kernel, result, Box(2.0), Box(3.0))
    assert result.value == 5.0

    routing.scale(kernel, result, 4.0, Box(2.0))
    assert result.value == 8.0

    routing.combine(kernel, result, [2.0, 3.0], [Box(10.0), Box(20.0)])
    assert result.value == 80.0


def test_vector_in_place_uses_in_place_translate():
    routing = RoutingVectorInPlace()
    kernel = InPlaceKernel()

    result = Box(np.zeros(2))
    origin = Box(np.array([1.0, 2.0]))
    delta = Box(np.array([3.0, 4.0]))

    original_result_value = result.value

    routing.translate(kernel, result, origin, delta)

    assert result.value is original_result_value
    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))


def test_vector_in_place_uses_in_place_add_scale_and_combine():
    routing = RoutingVectorInPlace()
    kernel = InPlaceKernel()

    result = Box(np.zeros(2))

    routing.add(kernel, result, Box(np.array([1.0, 2.0])), Box(np.array([3.0, 4.0])))
    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))

    routing.scale(kernel, result, 2.0, Box(np.array([3.0, 4.0])))
    np.testing.assert_allclose(result.value, np.array([6.0, 8.0]))

    routing.combine(
        kernel,
        result,
        [2.0, 3.0],
        [Box(np.array([1.0, 2.0])), Box(np.array([3.0, 4.0]))],
    )
    np.testing.assert_allclose(result.value, np.array([11.0, 16.0]))


def test_vector_in_place_fails_clearly_without_in_place_method():
    routing = RoutingVectorInPlace()
    kernel = ReturningKernel()

    with pytest.raises(AttributeError):
        routing.translate(kernel, Box(0.0), Box(1.0), Box(2.0))


def test_vector_prefer_in_place_uses_in_place_when_available():
    routing = RoutingVectorPreferInPlace()
    kernel = InPlaceKernel()

    result = Box(np.zeros(2))
    original_result_value = result.value

    routing.translate(
        kernel,
        result,
        Box(np.array([1.0, 2.0])),
        Box(np.array([3.0, 4.0])),
    )

    assert result.value is original_result_value
    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))


def test_vector_prefer_in_place_falls_back_to_returning_when_absent():
    routing = RoutingVectorPreferInPlace()
    kernel = ReturningKernel()

    result = Box(0.0)

    routing.translate(kernel, result, Box(1.0), Box(2.0))

    assert result.value == 3.0


def test_native_carrier_recommends_return_vector_routing():
    routing = CarrierNative().recommend_vector_routing()

    assert isinstance(routing, RoutingVectorReturn)


def test_numpy_carrier_recommends_prefer_in_place_vector_routing():
    routing = CarrierNumpy().recommend_vector_routing()

    assert isinstance(routing, RoutingVectorPreferInPlace)


def test_carrier_recommendation_can_build_routing():
    vector_routing = CarrierNumpy().recommend_vector_routing()

    routing = Routing(vector=vector_routing)

    assert routing.policy(RoutingVector) is vector_routing
