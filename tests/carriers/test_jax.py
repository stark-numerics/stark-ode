import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from stark.carriers.jax import (  # noqa: E402
    CarrierJax,
    CarrierKernelJax,
    CarrierNormJaxMax,
    CarrierNormJaxRMS,
)
from stark.carriers.library import CarrierLibrary  # noqa: E402
from stark.interface import StarkDerivative, StarkVector  # noqa: E402
from stark.interface.vector import StarkVectorTranslation  # noqa: E402
from stark.routing import RoutingVectorReturn  # noqa: E402


def test_jax_rms_norm():
    norm = CarrierNormJaxRMS().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
    )

    assert norm(jnp.array([3.0, 4.0])) == pytest.approx((12.5) ** 0.5)


def test_jax_max_norm():
    norm = CarrierNormJaxMax().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
    )

    assert norm(jnp.array([-3.0, 4.0])) == pytest.approx(4.0)


def test_jax_empty_array_norm():
    norm = CarrierNormJaxRMS().bind(
        template=jnp.array([]),
        carrier=None,
    )

    assert norm(jnp.array([])) == 0.0


def test_jax_complex_norm():
    norm = CarrierNormJaxRMS().bind(
        template=jnp.array([1.0 + 0.0j]),
        carrier=None,
    )

    result = norm(jnp.array([3.0 + 4.0j]))

    assert result == pytest.approx(5.0)


def test_jax_kernel_translate_add_scale_combine():
    norm = CarrierNormJaxRMS().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
    )
    kernel = CarrierKernelJax().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
        norm=norm,
    )

    assert jnp.allclose(
        kernel.translate(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
        jnp.array([4.0, 6.0]),
    )
    assert jnp.allclose(
        kernel.add(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
        jnp.array([4.0, 6.0]),
    )
    assert jnp.allclose(
        kernel.scale(2.0, jnp.array([3.0, 4.0])),
        jnp.array([6.0, 8.0]),
    )
    assert jnp.allclose(
        kernel.combine(
            [2.0, 3.0],
            [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])],
        ),
        jnp.array([11.0, 16.0]),
    )


def test_jax_kernel_norm_delegates_to_jax_norm():
    norm = CarrierNormJaxRMS().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
    )
    kernel = CarrierKernelJax().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
        norm=norm,
    )

    assert kernel.norm(jnp.array([3.0, 4.0])) == pytest.approx((12.5) ** 0.5)


def test_jax_kernel_empty_combine_raises():
    norm = CarrierNormJaxRMS().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
    )
    kernel = CarrierKernelJax().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
        norm=norm,
    )

    with pytest.raises(ValueError):
        kernel.combine([], [])


def test_jax_kernel_has_no_in_place_methods():
    kernel = CarrierKernelJax().bind(
        template=jnp.array([1.0, 2.0]),
        carrier=None,
        norm=lambda value: 0.0,
    )

    assert not hasattr(kernel, "translate_into")
    assert not hasattr(kernel, "add_into")
    assert not hasattr(kernel, "scale_into")
    assert not hasattr(kernel, "combine_into")


def test_carrier_jax_accepts_jax_array():
    assert CarrierJax().accepts(jnp.array([1.0, 2.0]))


def test_carrier_jax_binds_template():
    template = jnp.array([1.0, 2.0])
    carrier = CarrierJax().bind(template)

    assert carrier.template.shape == template.shape
    assert carrier.template.dtype == template.dtype


def test_carrier_jax_validates_shape():
    carrier = CarrierJax(strict_shape=True).bind(jnp.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        carrier.validate_state(jnp.array([1.0, 2.0, 3.0]))


def test_carrier_jax_validates_dtype_when_strict_dtype_is_true():
    carrier = CarrierJax(strict_dtype=True).bind(jnp.array([1.0, 2.0]))

    with pytest.raises(TypeError):
        carrier.validate_state(jnp.array([1, 2], dtype=jnp.int32))


def test_carrier_jax_recommends_return_routing():
    routing = CarrierJax().recommend_vector_routing()

    assert isinstance(routing, RoutingVectorReturn)


def test_carrier_library_default_selects_jax_for_jax_array():
    carrier = CarrierLibrary.default().carrier_for(jnp.array([1.0, 2.0]))

    assert isinstance(carrier, CarrierJax)


def test_stark_vector_translation_works_with_jax_return_routing():
    carrier = CarrierJax().bind(jnp.array([1.0, 2.0]))

    origin = StarkVector(jnp.array([1.0, 2.0]), carrier)
    original_result_value = jnp.zeros(2)
    result = StarkVector(original_result_value, carrier)
    translation = StarkVectorTranslation(
        jnp.array([3.0, 4.0]),
        carrier,
        RoutingVectorReturn(),
    )

    translation(origin, result)

    assert result.value is not original_result_value
    assert jnp.allclose(result.value, jnp.array([4.0, 6.0]))


def test_stark_derivative_return_convention_works_with_jax():
    carrier = CarrierJax().bind(jnp.array([1.0, 2.0]))

    @StarkDerivative.returning
    def rhs(t, y):
        return t * y

    bound = rhs.bind(carrier)

    class Interval:
        present = 2.0

    state = StarkVector(jnp.array([3.0, 4.0]), carrier)
    out = StarkVectorTranslation(jnp.zeros(2), carrier)

    bound(Interval(), state, out)

    assert jnp.allclose(out.value, jnp.array([6.0, 8.0]))
