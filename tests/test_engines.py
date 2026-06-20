from __future__ import annotations

import numpy as np
import pytest

from stark import Frame, FrameField
from stark.engines.shared.accelerators import AcceleratorNone
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.numpy.carriers import CarrierNumpy
from stark.engines import EngineNative, EngineNumpy


def test_stark_engine_numpy_exposes_backend_bundle() -> None:
    accelerator = AcceleratorNone()
    layout = Frame(
        (
            FrameField("u", translation="du", shape=(2, 3)),
            FrameField("v", translation="dv", shape=(2, 3)),
        )
    )

    engine = EngineNumpy(layout, accelerator=accelerator)

    assert engine.frame is layout
    assert engine.accelerator is accelerator
    assert len(engine.carriers) == 2
    assert all(isinstance(carrier, CarrierNumpy) for carrier in engine.carriers)
    assert tuple(str(path) for path in engine.algebraist_frame.state_paths) == ("u", "v")


def test_stark_engine_numpy_allocator_builds_owned_structures() -> None:
    engine = EngineNumpy(
        Frame(
            (
                FrameField("u", translation="du", shape=(2, 2)),
                FrameField("v", translation="dv", shape=(2, 2)),
            )
        ),
        dtype=np.float32,
    )

    state = engine.allocator.allocate_state()
    translation = engine.allocator.allocate_translation()

    assert state.u.shape == (2, 2)
    assert state.v.dtype == np.float32
    assert translation.du.shape == (2, 2)
    assert translation.dv.dtype == np.float32


def test_stark_engine_numpy_translation_applies_fieldwise_delta() -> None:
    engine = EngineNumpy(
        Frame(
            (
                FrameField("u", translation="du", shape=(2,)),
                FrameField("v", translation="dv", shape=(2,)),
            )
        )
    )
    origin = engine.allocator.allocate_state()
    result = engine.allocator.allocate_state()
    delta = engine.allocator.allocate_translation()

    origin.u[...] = (1.0, 2.0)
    origin.v[...] = (3.0, 4.0)
    delta.du[...] = (10.0, 20.0)
    delta.dv[...] = (30.0, 40.0)

    delta(origin, result)

    np.testing.assert_array_equal(result.u, np.array([11.0, 22.0]))
    np.testing.assert_array_equal(result.v, np.array([33.0, 44.0]))
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)


def test_stark_engine_numpy_exposes_algebraist_providers() -> None:
    engine = EngineNumpy(
        Frame(
            (
                FrameField("u", translation="du", shape=(2,)),
                FrameField("v", translation="dv", shape=(2,)),
            )
        )
    )

    assert isinstance(engine.algebraist_linear_combine, AlgebraistGeneratorLinearCombine)
    assert isinstance(engine.algebraist_inner_product, AlgebraistGeneratorInnerProduct)
    assert isinstance(engine.algebraist_norm, AlgebraistGeneratorNorm)
    assert isinstance(engine.algebraist_specialist, AlgebraistGeneratorSpecialist)


def test_stark_engine_numpy_exposes_layout_inner_product() -> None:
    engine = EngineNumpy(
        Frame(
            (
                FrameField("u", translation="du", shape=(2,)),
                FrameField("v", translation="dv", shape=(2,)),
            )
        )
    )
    left = engine.allocator.allocate_translation()
    right = engine.allocator.allocate_translation()

    left.du[...] = (1.0, 2.0)
    left.dv[...] = (3.0, 4.0)
    right.du[...] = (10.0, 20.0)
    right.dv[...] = (30.0, 40.0)

    assert engine.allocator.inner_product(left, right) == pytest.approx(150.0)


def test_stark_engine_native_uses_array_backed_fields() -> None:
    engine = EngineNative(
        Frame(
            (
                FrameField("u", translation="du", shape=(2,)),
                FrameField("v", translation="dv", shape=(2,)),
            )
        )
    )
    origin = engine.allocator.allocate_state()
    result = engine.allocator.allocate_state()
    delta = engine.allocator.allocate_translation()

    origin.u[0], origin.u[1] = 1.0, 2.0
    origin.v[0], origin.v[1] = 3.0, 4.0
    delta.du[0], delta.du[1] = 10.0, 20.0
    delta.dv[0], delta.dv[1] = 30.0, 40.0

    delta(origin, result)

    assert list(result.u) == [11.0, 22.0]
    assert list(result.v) == [33.0, 44.0]
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)
    assert isinstance(engine.algebraist_linear_combine, AlgebraistGeneratorLinearCombine)
    assert isinstance(engine.algebraist_inner_product, AlgebraistGeneratorInnerProduct)
    assert isinstance(engine.algebraist_norm, AlgebraistGeneratorNorm)
    assert isinstance(engine.algebraist_specialist, AlgebraistGeneratorSpecialist)
    assert engine.allocator.inner_product(delta, delta) == pytest.approx(1500.0)


def test_stark_engine_native_rejects_multidimensional_shapes() -> None:
    layout = Frame((FrameField("u", translation="du", shape=(2, 2)),))

    with pytest.raises(ValueError, match="one-dimensional"):
        EngineNative(layout)


def test_stark_engine_cupy_optional() -> None:
    cp = pytest.importorskip("cupy")
    from stark.engines.cupy import EngineCupy

    engine = EngineCupy(
        Frame(
            (
                FrameField("u", translation="du", shape=(2,)),
                FrameField("v", translation="dv", shape=(2,)),
            )
        ),
        dtype=cp.float32,
    )
    origin = engine.allocator.allocate_state()
    result = engine.allocator.allocate_state()
    delta = engine.allocator.allocate_translation()

    origin.u[...] = cp.asarray((1.0, 2.0))
    origin.v[...] = cp.asarray((3.0, 4.0))
    delta.du[...] = cp.asarray((10.0, 20.0))
    delta.dv[...] = cp.asarray((30.0, 40.0))

    delta(origin, result)

    np.testing.assert_array_equal(cp.asnumpy(result.u), np.array([11.0, 22.0]))
    np.testing.assert_array_equal(cp.asnumpy(result.v), np.array([33.0, 44.0]))
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)
    assert isinstance(engine.algebraist_linear_combine, AlgebraistGeneratorLinearCombine)
    assert isinstance(engine.algebraist_inner_product, AlgebraistGeneratorInnerProduct)
    assert isinstance(engine.algebraist_norm, AlgebraistGeneratorNorm)
    assert isinstance(engine.algebraist_specialist, AlgebraistGeneratorSpecialist)
    assert len(engine.allocator.linear_combine) >= 2
    inner_product = engine.allocator.inner_product(delta, delta)
    assert float(inner_product.get()) == pytest.approx(1500.0)


def test_stark_engine_jax_optional() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from stark.engines.jax import EngineJax

    engine = EngineJax(
        Frame(
            (
                FrameField("u", translation="du", shape=(2,)),
                FrameField("v", translation="dv", shape=(2,)),
            )
        ),
        dtype=jnp.float32,
    )
    origin = engine.allocator.allocate_state()
    result = engine.allocator.allocate_state()
    delta = engine.allocator.allocate_translation()

    origin.u = jnp.asarray((1.0, 2.0))
    origin.v = jnp.asarray((3.0, 4.0))
    delta.du = jnp.asarray((10.0, 20.0))
    delta.dv = jnp.asarray((30.0, 40.0))

    delta(origin, result)

    np.testing.assert_array_equal(np.asarray(result.u), np.array([11.0, 22.0]))
    np.testing.assert_array_equal(np.asarray(result.v), np.array([33.0, 44.0]))
    assert delta.norm() == pytest.approx((250.0 + 1250.0) ** 0.5)
    assert isinstance(engine.algebraist_linear_combine, AlgebraistGeneratorLinearCombine)
    assert isinstance(engine.algebraist_inner_product, AlgebraistGeneratorInnerProduct)
    assert isinstance(engine.algebraist_norm, AlgebraistGeneratorNorm)
    assert isinstance(engine.algebraist_specialist, AlgebraistGeneratorSpecialist)
    assert len(engine.allocator.linear_combine) >= 2
    assert engine.allocator.inner_product(delta, delta) == pytest.approx(1500.0)
