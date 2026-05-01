import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from stark import Interval  # noqa: E402
from stark.carriers.jax import CarrierJax  # noqa: E402
from stark.interface import StarkDerivative, StarkIVP, StarkVector  # noqa: E402
from stark.interface.vector import StarkVectorTranslation  # noqa: E402
from stark.routing import RoutingVectorReturn  # noqa: E402


def test_jax_interface_smoke_state_translation_and_derivative():
    carrier = CarrierJax().bind(jnp.array([1.0, 2.0]))

    state = StarkVector(jnp.array([1.0, 2.0]), carrier)
    result = StarkVector(jnp.zeros(2), carrier)
    translation = StarkVectorTranslation(
        jnp.array([3.0, 4.0]),
        carrier,
        RoutingVectorReturn(),
    )

    translation(state, result)

    assert jnp.allclose(result.value, jnp.array([4.0, 6.0]))

    @StarkDerivative.returning
    def rhs(t, y):
        return t * y

    bound = rhs.bind(carrier)
    out = StarkVectorTranslation(jnp.zeros(2), carrier)

    class DummyInterval:
        present = 2.0

    bound(DummyInterval(), state, out)

    assert jnp.allclose(out.value, jnp.array([2.0, 4.0]))


def test_stark_ivp_prepares_jax_initial_with_jax_carrier_and_return_routing():
    initial = jnp.array([1.0, 2.0])

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=initial,
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    assert isinstance(ivp.prepared_carrier.carrier, CarrierJax)
    assert isinstance(ivp.vector_routing, RoutingVectorReturn)
    assert jnp.allclose(ivp.prepared_initial.value, initial)


def test_stark_ivp_jax_python_level_integrate_smoke():
    initial = jnp.array([1.0, 2.0])

    ivp = StarkIVP(
        derivative=lambda t, y: -y,
        initial=initial,
        interval=Interval(present=0.0, step=0.1, stop=0.2),
    )

    results = list(ivp.integrate())

    assert results

    final_interval, final_state = results[-1]

    assert final_interval.present == pytest.approx(0.2)
    assert jnp.all(final_state.value > 0.0)
    assert jnp.all(final_state.value < initial)