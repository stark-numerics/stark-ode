import numpy as np
import pytest

from stark.carriers import CarrierNative, CarrierNumpy
from stark.conventions import ConventionReturn, ConventionInPlace


def test_return_convention_calls_function_with_t_and_y():
    convention = ConventionReturn()
    carrier = CarrierNative().bind([1.0, 2.0])

    calls = []

    def rhs(t, y):
        calls.append((t, y))
        return [2.0, 4.0]

    result = convention(rhs, 1.5, [1.0, 2.0], [0.0, 0.0], carrier)

    assert calls == [(1.5, [1.0, 2.0])]
    assert result == [2.0, 4.0]


def test_return_convention_coerces_native_output():
    convention = ConventionReturn()
    carrier = CarrierNative().bind([1.0, 2.0])

    def rhs(t, y):
        return [1, 2]

    result = convention(rhs, 0.0, [1.0, 2.0], [0.0, 0.0], carrier)

    assert result == [1.0, 2.0]


def test_return_convention_coerces_numpy_output():
    convention = ConventionReturn()
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))

    def rhs(t, y):
        return [3.0, 4.0]

    result = convention(rhs, 0.0, np.array([1.0, 2.0]), np.zeros(2), carrier)

    np.testing.assert_allclose(result, np.array([3.0, 4.0]))


def test_return_convention_rejects_wrong_numpy_shape():
    convention = ConventionReturn()
    carrier = CarrierNumpy(strict_shape=True).bind(np.array([1.0, 2.0]))

    def rhs(t, y):
        return np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        convention(rhs, 0.0, np.array([1.0, 2.0]), np.zeros(2), carrier)


def test_in_place_convention_calls_function_with_t_y_dy():
    convention = ConventionInPlace()
    carrier = CarrierNative().bind([1.0, 2.0])

    calls = []

    def rhs(t, y, dy):
        calls.append((t, y, dy))
        dy[0] = 5.0
        dy[1] = 6.0

    dy = [0.0, 0.0]

    result = convention(rhs, 2.0, [1.0, 2.0], dy, carrier)

    assert calls == [(2.0, [1.0, 2.0], [5.0, 6.0])]
    assert result == [5.0, 6.0]


def test_in_place_convention_validates_native_output():
    convention = ConventionInPlace()
    carrier = CarrierNative().bind([1.0, 2.0])

    def rhs(t, y, dy):
        dy.append(3.0)

    with pytest.raises(ValueError):
        convention(rhs, 0.0, [1.0, 2.0], [0.0, 0.0], carrier)


def test_in_place_convention_validates_numpy_output():
    convention = ConventionInPlace()
    carrier = CarrierNumpy(strict_shape=True).bind(np.array([1.0, 2.0]))

    def rhs(t, y, dy):
        dy[:] = np.array([3.0, 4.0])

    dy = np.zeros(2)

    result = convention(rhs, 0.0, np.array([1.0, 2.0]), dy, carrier)

    assert result is dy
    np.testing.assert_allclose(result, np.array([3.0, 4.0]))


def test_in_place_convention_rejects_wrong_numpy_shape():
    convention = ConventionInPlace()
    carrier = CarrierNumpy(strict_shape=True).bind(np.array([1.0, 2.0]))

    def rhs(t, y, dy):
        pass

    with pytest.raises(ValueError):
        convention(rhs, 0.0, np.array([1.0, 2.0]), np.zeros(3), carrier)