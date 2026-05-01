import pytest

cp = pytest.importorskip("cupy")

from stark.carriers.cupy import (  # noqa: E402
    CarrierCuPy,
    CarrierKernelCuPy,
    CarrierNormCuPyMax,
    CarrierNormCuPyRMS,
)
from stark.carriers.library import CarrierLibrary  # noqa: E402
from stark.interface import StarkDerivative, StarkVector  # noqa: E402
from stark.interface.vector import StarkVectorTranslation  # noqa: E402
from stark.routing import RoutingVectorPreferInPlace  # noqa: E402


def require_working_cupy():
    try:
        cp.zeros(1)
    except Exception as error:
        pytest.skip(f"CuPy is installed but not usable in this environment: {error}")


def test_cupy_rms_norm():
    require_working_cupy()

    norm = CarrierNormCuPyRMS().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
    )

    assert norm(cp.array([3.0, 4.0])) == pytest.approx((12.5) ** 0.5)


def test_cupy_max_norm():
    require_working_cupy()

    norm = CarrierNormCuPyMax().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
    )

    assert norm(cp.array([-3.0, 4.0])) == pytest.approx(4.0)


def test_cupy_empty_array_norm():
    require_working_cupy()

    norm = CarrierNormCuPyRMS().bind(
        template=cp.array([]),
        carrier=None,
    )

    assert norm(cp.array([])) == 0.0


def test_cupy_complex_norm():
    require_working_cupy()

    norm = CarrierNormCuPyRMS().bind(
        template=cp.array([1.0 + 0.0j]),
        carrier=None,
    )

    result = norm(cp.array([3.0 + 4.0j]))

    assert result == pytest.approx(5.0)


def test_cupy_kernel_translate_add_scale_combine():
    require_working_cupy()

    norm = CarrierNormCuPyRMS().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
    )
    kernel = CarrierKernelCuPy().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
        norm=norm,
    )

    cp.testing.assert_allclose(
        kernel.translate(cp.array([1.0, 2.0]), cp.array([3.0, 4.0])),
        cp.array([4.0, 6.0]),
    )
    cp.testing.assert_allclose(
        kernel.add(cp.array([1.0, 2.0]), cp.array([3.0, 4.0])),
        cp.array([4.0, 6.0]),
    )
    cp.testing.assert_allclose(
        kernel.scale(2.0, cp.array([3.0, 4.0])),
        cp.array([6.0, 8.0]),
    )
    cp.testing.assert_allclose(
        kernel.combine(
            [2.0, 3.0],
            [cp.array([1.0, 2.0]), cp.array([3.0, 4.0])],
        ),
        cp.array([11.0, 16.0]),
    )


def test_cupy_kernel_in_place_methods():
    require_working_cupy()

    norm = CarrierNormCuPyRMS().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
    )
    kernel = CarrierKernelCuPy().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
        norm=norm,
    )

    result = cp.zeros(2)

    kernel.translate_into(result, cp.array([1.0, 2.0]), cp.array([3.0, 4.0]))
    cp.testing.assert_allclose(result, cp.array([4.0, 6.0]))

    kernel.add_into(result, cp.array([1.0, 2.0]), cp.array([3.0, 4.0]))
    cp.testing.assert_allclose(result, cp.array([4.0, 6.0]))

    kernel.scale_into(result, 2.0, cp.array([3.0, 4.0]))
    cp.testing.assert_allclose(result, cp.array([6.0, 8.0]))

    kernel.combine_into(
        result,
        [2.0, 3.0],
        [cp.array([1.0, 2.0]), cp.array([3.0, 4.0])],
    )
    cp.testing.assert_allclose(result, cp.array([11.0, 16.0]))


def test_cupy_kernel_empty_combine_raises():
    require_working_cupy()

    norm = CarrierNormCuPyRMS().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
    )
    kernel = CarrierKernelCuPy().bind(
        template=cp.array([1.0, 2.0]),
        carrier=None,
        norm=norm,
    )

    with pytest.raises(ValueError):
        kernel.combine([], [])


def test_carrier_cupy_accepts_cupy_array():
    require_working_cupy()

    assert CarrierCuPy().accepts(cp.array([1.0, 2.0]))


def test_carrier_cupy_binds_template():
    require_working_cupy()

    template = cp.array([1.0, 2.0])
    carrier = CarrierCuPy().bind(template)

    assert carrier.template.shape == template.shape
    assert carrier.template.dtype == template.dtype


def test_carrier_cupy_validates_shape():
    require_working_cupy()

    carrier = CarrierCuPy(strict_shape=True).bind(cp.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        carrier.validate_state(cp.array([1.0, 2.0, 3.0]))


def test_carrier_cupy_validates_dtype_when_strict_dtype_is_true():
    require_working_cupy()

    carrier = CarrierCuPy(strict_dtype=True).bind(cp.array([1.0, 2.0]))

    with pytest.raises(TypeError):
        carrier.validate_state(cp.array([1, 2], dtype=cp.int64))


def test_carrier_cupy_recommends_prefer_in_place_routing():
    require_working_cupy()

    routing = CarrierCuPy().recommend_vector_routing()

    assert isinstance(routing, RoutingVectorPreferInPlace)


def test_carrier_library_default_selects_cupy_for_cupy_array():
    require_working_cupy()

    carrier = CarrierLibrary.default().carrier_for(cp.array([1.0, 2.0]))

    assert isinstance(carrier, CarrierCuPy)


def test_stark_vector_translation_works_with_cupy():
    require_working_cupy()

    carrier = CarrierCuPy().bind(cp.array([1.0, 2.0]))

    origin = StarkVector(cp.array([1.0, 2.0]), carrier)
    result = StarkVector(cp.zeros(2), carrier)
    translation = StarkVectorTranslation(
        cp.array([3.0, 4.0]),
        carrier,
        RoutingVectorPreferInPlace(),
    )

    translation(origin, result)

    cp.testing.assert_allclose(result.value, cp.array([4.0, 6.0]))


def test_stark_derivative_return_convention_works_with_cupy():
    require_working_cupy()

    carrier = CarrierCuPy().bind(cp.array([1.0, 2.0]))

    @StarkDerivative.returning
    def rhs(t, y):
        return t * y

    bound = rhs.bind(carrier)

    class Interval:
        present = 2.0

    state = StarkVector(cp.array([3.0, 4.0]), carrier)
    out = StarkVectorTranslation(cp.zeros(2), carrier)

    bound(Interval(), state, out)

    cp.testing.assert_allclose(out.value, cp.array([6.0, 8.0]))
