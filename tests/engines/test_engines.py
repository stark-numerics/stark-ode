from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pytest

from stark import Frame, Field
from stark.engines.accelerators import AcceleratorNone
from stark.engines.carrier_numpy import CarrierNumpy
from stark.engines.generator import (
    Generator,
    GeneratorInnerProduct,
    GeneratorLinearFixed,
    GeneratorNorm,
)
from stark.engines import EngineNative, EngineNumpy


class EnginePairState(Protocol):
    """Runtime state shape produced by the two-field engine tests."""

    u: Any
    v: Any


class EnginePairTranslation(Protocol):
    """Runtime translation shape produced by the two-field engine tests."""

    du: Any
    dv: Any

    def __call__(self, origin: EnginePairState, result: EnginePairState) -> None:
        ...

    def norm(self) -> float:
        ...


def allocate_pair_state(engine: Any) -> EnginePairState:
    """Allocate a two-field test state with protocol-visible attributes."""

    return engine.allocator.allocate_state()


def allocate_pair_translation(engine: Any) -> EnginePairTranslation:
    """Allocate a two-field test translation with protocol-visible attributes."""

    return engine.allocator.allocate_translation()


def test_stark_engine_numpy_exposes_backend_bundle() -> None:
    accelerator = AcceleratorNone()
    layout = Frame(
        (
            Field("u", translation="du", shape=(2, 3)),
            Field("v", translation="dv", shape=(2, 3)),
        )
    )

    engine = EngineNumpy(layout, accelerator=accelerator)

    assert engine.frame is layout
    assert engine.accelerator is accelerator
    assert len(engine.carriers) == 2
    assert all(isinstance(carrier, CarrierNumpy) for carrier in engine.carriers)
    assert tuple(str(path) for path in engine.frame.state_paths) == ("u", "v")


def test_stark_engine_numpy_allocator_builds_owned_structures() -> None:
    engine = EngineNumpy(
        Frame(
            (
                Field("u", translation="du", shape=(2, 2)),
                Field("v", translation="dv", shape=(2, 2)),
            )
        ),
        dtype=np.float32,
    )

    state = allocate_pair_state(engine)
    translation = allocate_pair_translation(engine)

    assert state.u.shape == (2, 2)
    assert state.v.dtype == np.float32
    assert translation.du.shape == (2, 2)
    assert translation.dv.dtype == np.float32


def test_stark_engine_numpy_translation_applies_fieldwise_delta() -> None:
    engine = EngineNumpy(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(2,)),
            )
        )
    )
    origin = allocate_pair_state(engine)
    result = allocate_pair_state(engine)
    delta = allocate_pair_translation(engine)

    origin.u[...] = (1.0, 2.0)
    origin.v[...] = (3.0, 4.0)
    delta.du[...] = (10.0, 20.0)
    delta.dv[...] = (30.0, 40.0)

    delta(origin, result)

    np.testing.assert_array_equal(result.u, np.array([11.0, 22.0]))
    np.testing.assert_array_equal(result.v, np.array([33.0, 44.0]))
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)


def test_stark_engine_numpy_exposes_generator_providers() -> None:
    engine = EngineNumpy(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(2,)),
            )
        )
    )

    assert isinstance(engine.generator, Generator)
    assert isinstance(engine.generator.inner_product, GeneratorInnerProduct)
    assert isinstance(engine.generator.linear_fixed, GeneratorLinearFixed)
    assert isinstance(engine.generator.norm, GeneratorNorm)
    assert len(engine.allocator.linear_combine) >= 2


def test_stark_engine_numpy_exposes_layout_inner_product() -> None:
    engine = EngineNumpy(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(2,)),
            )
        )
    )
    left = allocate_pair_translation(engine)
    right = allocate_pair_translation(engine)

    left.du[...] = (1.0, 2.0)
    left.dv[...] = (3.0, 4.0)
    right.du[...] = (10.0, 20.0)
    right.dv[...] = (30.0, 40.0)

    inner_product = engine.allocator.inner_product
    assert inner_product is not None
    assert inner_product(left, right) == pytest.approx(300.0)


def test_stark_engine_numpy_translation_basis_inspects_full_translation() -> None:
    engine = EngineNumpy(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(1,)),
            )
        )
    )
    basis = engine.translation_basis()
    vector = allocate_pair_translation(engine)
    translation = allocate_pair_translation(engine)
    coordinates = [0.0, 0.0, 0.0]

    vector = basis.vector(1, vector)
    assert basis.dimension == 3
    np.testing.assert_array_equal(vector.du, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(vector.dv, np.array([0.0]))
    assert basis.coordinate(1, vector) == pytest.approx(1.0)
    assert basis.coordinate(2, vector) == pytest.approx(0.0)

    translation.du[...] = (3.0, 4.0)
    translation.dv[...] = (5.0,)
    basis.coordinates(translation, coordinates)
    assert coordinates == pytest.approx([3.0, 4.0, 5.0])

    basis.synthesize([6.0, 7.0, 8.0], translation)
    np.testing.assert_array_equal(translation.du, np.array([6.0, 7.0]))
    np.testing.assert_array_equal(translation.dv, np.array([8.0]))


def test_stark_engine_native_uses_array_backed_fields() -> None:
    engine = EngineNative(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(2,)),
            )
        )
    )
    origin = allocate_pair_state(engine)
    result = allocate_pair_state(engine)
    delta = allocate_pair_translation(engine)

    origin.u[0], origin.u[1] = 1.0, 2.0
    origin.v[0], origin.v[1] = 3.0, 4.0
    delta.du[0], delta.du[1] = 10.0, 20.0
    delta.dv[0], delta.dv[1] = 30.0, 40.0

    delta(origin, result)

    assert list(result.u) == [11.0, 22.0]
    assert list(result.v) == [33.0, 44.0]
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)
    assert isinstance(engine.generator, Generator)
    assert isinstance(engine.generator.inner_product, GeneratorInnerProduct)
    assert isinstance(engine.generator.linear_fixed, GeneratorLinearFixed)
    assert isinstance(engine.generator.norm, GeneratorNorm)
    assert len(engine.allocator.linear_combine) >= 2
    inner_product = engine.allocator.inner_product
    assert inner_product is not None
    assert inner_product(delta, delta) == pytest.approx(3000.0)


def test_stark_engine_native_rejects_multidimensional_shapes() -> None:
    layout = Frame((Field("u", translation="du", shape=(2, 2)),))

    with pytest.raises(ValueError, match="one-dimensional"):
        EngineNative(layout)


def test_stark_engine_cupy_optional() -> None:
    cp = pytest.importorskip("cupy")
    from stark.engines.engine_cupy import EngineCupy

    engine = EngineCupy(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(2,)),
            )
        ),
        dtype=cp.float32,
    )
    origin = allocate_pair_state(engine)
    result = allocate_pair_state(engine)
    delta = allocate_pair_translation(engine)

    origin.u[...] = cp.asarray((1.0, 2.0))
    origin.v[...] = cp.asarray((3.0, 4.0))
    delta.du[...] = cp.asarray((10.0, 20.0))
    delta.dv[...] = cp.asarray((30.0, 40.0))

    delta(origin, result)

    np.testing.assert_array_equal(cp.asnumpy(result.u), np.array([11.0, 22.0]))
    np.testing.assert_array_equal(cp.asnumpy(result.v), np.array([33.0, 44.0]))
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)
    assert isinstance(engine.generator, Generator)
    assert isinstance(engine.generator.inner_product, GeneratorInnerProduct)
    assert isinstance(engine.generator.linear_fixed, GeneratorLinearFixed)
    assert isinstance(engine.generator.norm, GeneratorNorm)
    assert len(engine.allocator.linear_combine) >= 2
    inner_product = engine.allocator.inner_product
    assert inner_product is not None
    inner_product_value: Any = inner_product(delta, delta)
    assert float(inner_product_value.get()) == pytest.approx(3000.0)


def test_stark_engine_jax_optional() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from stark.engines.engine_jax import EngineJax

    engine = EngineJax(
        Frame(
            (
                Field("u", translation="du", shape=(2,)),
                Field("v", translation="dv", shape=(2,)),
            )
        ),
        dtype=jnp.float32,
    )
    origin = allocate_pair_state(engine)
    result = allocate_pair_state(engine)
    delta = allocate_pair_translation(engine)

    origin.u = jnp.asarray((1.0, 2.0))
    origin.v = jnp.asarray((3.0, 4.0))
    delta.du = jnp.asarray((10.0, 20.0))
    delta.dv = jnp.asarray((30.0, 40.0))

    delta(origin, result)

    np.testing.assert_array_equal(np.asarray(result.u), np.array([11.0, 22.0]))
    np.testing.assert_array_equal(np.asarray(result.v), np.array([33.0, 44.0]))
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)
    assert isinstance(engine.generator, Generator)
    assert isinstance(engine.generator.inner_product, GeneratorInnerProduct)
    assert isinstance(engine.generator.linear_fixed, GeneratorLinearFixed)
    assert isinstance(engine.generator.norm, GeneratorNorm)
    assert len(engine.allocator.linear_combine) >= 2
    inner_product = engine.allocator.inner_product
    assert inner_product is not None
    assert inner_product(delta, delta) == pytest.approx(3000.0)
