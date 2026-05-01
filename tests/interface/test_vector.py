import numpy as np

from stark.carriers import CarrierNative, CarrierNumpy
from stark.interface import StarkVector
from stark.interface.vector import StarkVectorTranslation, StarkVectorWorkbench
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.routing import RoutingVectorPreferInPlace, RoutingVectorReturn


def test_stark_vector_stores_native_value_and_carrier():
    carrier = CarrierNative().bind([1.0, 2.0])
    value = [1.0, 2.0]

    vector = StarkVector(value, carrier)

    assert vector.value == value
    assert vector.carrier is carrier


def test_stark_vector_stores_numpy_value_and_carrier():
    value = np.array([1.0, 2.0])
    carrier = CarrierNumpy().bind(value)

    vector = StarkVector(value, carrier)

    assert vector.value is value
    assert vector.carrier is carrier


def test_native_translation_applies_to_state():
    carrier = CarrierNative().bind([1.0, 2.0])

    origin = StarkVector([1.0, 2.0], carrier)
    result = StarkVector([0.0, 0.0], carrier)
    translation = StarkVectorTranslation([3.0, 4.0], carrier)

    returned = translation(origin, result)

    assert returned is result
    assert result.value == [4.0, 6.0]


def test_native_translation_norm_works():
    carrier = CarrierNative().bind([1.0, 2.0])
    translation = StarkVectorTranslation([3.0, 4.0], carrier)

    assert translation.norm() == (12.5 ** 0.5)


def test_native_translation_addition_works():
    carrier = CarrierNative().bind([1.0, 2.0])

    left = StarkVectorTranslation([1.0, 2.0], carrier)
    right = StarkVectorTranslation([3.0, 4.0], carrier)

    result = left + right

    assert result.value == [4.0, 6.0]
    assert result.carrier is carrier


def test_native_translation_scalar_multiplication_works():
    carrier = CarrierNative().bind([1.0, 2.0])
    translation = StarkVectorTranslation([3.0, 4.0], carrier)

    result = 2.0 * translation

    assert result.value == [6.0, 8.0]
    assert result.carrier is carrier


def test_numpy_translation_applies_to_state_using_return_routing():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))

    origin = StarkVector(np.array([1.0, 2.0]), carrier)
    original_result_value = np.zeros(2)
    result = StarkVector(original_result_value, carrier)
    translation = StarkVectorTranslation(
        np.array([3.0, 4.0]),
        carrier,
        RoutingVectorReturn(),
    )

    translation(origin, result)

    assert result.value is not original_result_value
    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))


def test_numpy_translation_applies_to_state_using_prefer_in_place_routing():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))

    origin = StarkVector(np.array([1.0, 2.0]), carrier)
    original_result_value = np.zeros(2)
    result = StarkVector(original_result_value, carrier)
    translation = StarkVectorTranslation(
        np.array([3.0, 4.0]),
        carrier,
        RoutingVectorPreferInPlace(),
    )

    translation(origin, result)

    assert result.value is original_result_value
    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))

def test_numpy_translation_norm_works():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))
    translation = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)

    assert translation.norm() == (12.5 ** 0.5)


def test_numpy_translation_addition_works():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))

    left = StarkVectorTranslation(np.array([1.0, 2.0]), carrier)
    right = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)

    result = left + right

    np.testing.assert_allclose(result.value, np.array([4.0, 6.0]))
    assert result.carrier is carrier


def test_numpy_translation_scalar_multiplication_works():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))
    translation = StarkVectorTranslation(np.array([3.0, 4.0]), carrier)

    result = 2.0 * translation

    np.testing.assert_allclose(result.value, np.array([6.0, 8.0]))
    assert result.carrier is carrier


def test_translation_exposes_linear_combine_through_combine12():
    carrier = CarrierNative().bind([0.0])
    translation = StarkVectorTranslation([0.0], carrier)

    assert len(translation.linear_combine) == 12

    out = StarkVectorTranslation([0.0], carrier)
    values = [
        StarkVectorTranslation([float(index)], carrier)
        for index in range(1, 13)
    ]
    terms = []
    for index, value in enumerate(values, start=1):
        terms.extend([float(index), value])

    combined = translation.combine12(out, *terms)

    assert combined is out
    assert out.value == [650.0]


def test_numpy_linear_combine_prefers_in_place_output_storage():
    carrier = CarrierNumpy().bind(np.array([0.0, 0.0]))
    routing = RoutingVectorPreferInPlace()
    translation = StarkVectorTranslation(np.array([0.0, 0.0]), carrier, routing)
    out_value = np.zeros(2)
    out = StarkVectorTranslation(out_value, carrier, routing)
    left = StarkVectorTranslation(np.array([1.0, 2.0]), carrier, routing)
    right = StarkVectorTranslation(np.array([3.0, 4.0]), carrier, routing)

    combined = translation.combine2(out, 2.0, left, 3.0, right)

    assert combined is out
    assert out.value is out_value
    np.testing.assert_allclose(out.value, np.array([11.0, 16.0]))


def test_scheme_workspace_consumes_stark_vector_linear_combine():
    carrier = CarrierNative().bind([0.0])
    workbench = StarkVectorWorkbench(carrier)
    probe = workbench.allocate_translation()
    workspace = SchemeWorkspace(workbench, probe)
    out = workbench.allocate_translation()
    values = [
        StarkVectorTranslation([float(index)], carrier)
        for index in range(1, 13)
    ]
    terms = []
    for index, value in enumerate(values, start=1):
        terms.extend([float(index), value])

    combined = workspace.combine12(out, *terms)

    assert combined is out
    assert out.value == [650.0]


def test_workbench_allocates_native_state():
    carrier = CarrierNative().bind([1.0, 2.0])
    workbench = StarkVectorWorkbench(carrier)

    state = workbench.allocate_state()

    assert isinstance(state, StarkVector)
    assert state.value == [0.0, 0.0]
    assert state.carrier is carrier


def test_workbench_copies_native_state():
    carrier = CarrierNative().bind([1.0, 2.0])
    workbench = StarkVectorWorkbench(carrier)

    source = StarkVector([3.0, 4.0], carrier)
    result = StarkVector([0.0, 0.0], carrier)

    returned = workbench.copy_state(result, source)

    assert returned is result
    assert result.value == [3.0, 4.0]


def test_workbench_allocates_native_translation():
    carrier = CarrierNative().bind([1.0, 2.0])
    workbench = StarkVectorWorkbench(carrier)

    translation = workbench.allocate_translation()

    assert isinstance(translation, StarkVectorTranslation)
    assert translation.value == [0.0, 0.0]
    assert translation.carrier is carrier


def test_workbench_allocates_numpy_state():
    template = np.array([1.0, 2.0])
    carrier = CarrierNumpy().bind(template)
    workbench = StarkVectorWorkbench(carrier)

    state = workbench.allocate_state()

    assert isinstance(state, StarkVector)
    np.testing.assert_allclose(state.value, np.array([0.0, 0.0]))
    assert state.carrier is carrier


def test_workbench_copies_numpy_state():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))
    workbench = StarkVectorWorkbench(carrier)

    source = StarkVector(np.array([3.0, 4.0]), carrier)
    result = StarkVector(np.zeros(2), carrier)

    returned = workbench.copy_state(result, source)

    assert returned is result
    np.testing.assert_allclose(result.value, np.array([3.0, 4.0]))
    assert result.value is not source.value


def test_workbench_allocates_numpy_translation():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))
    workbench = StarkVectorWorkbench(carrier)

    translation = workbench.allocate_translation()

    assert isinstance(translation, StarkVectorTranslation)
    np.testing.assert_allclose(translation.value, np.array([0.0, 0.0]))
    assert translation.carrier is carrier


def test_workbench_uses_routing_when_constructing_translations():
    carrier = CarrierNumpy().bind(np.array([1.0, 2.0]))
    routing = RoutingVectorPreferInPlace()
    workbench = StarkVectorWorkbench(carrier, routing)

    translation = workbench.allocate_translation()

    assert translation.routing is routing
