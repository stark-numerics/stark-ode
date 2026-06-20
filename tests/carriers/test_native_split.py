import inspect

from stark.engines.native.carriers import (
    CarrierArithmeticNativeList,
    CarrierArithmeticNativeScalar,
    CarrierArithmeticNativeTuple,
    CarrierNative,
    CarrierNativeList,
    CarrierNativeScalar,
    CarrierNativeTuple,
    CarrierStorageNativeScalar,
    CarrierStorageNativeList,
    CarrierStorageNativeTuple
)


def test_native_facade_selects_scalar_carrier() -> None:
    carrier = CarrierNative(2.0)

    assert isinstance(carrier.storage, CarrierStorageNativeScalar)
    assert isinstance(carrier.arithmetic, CarrierArithmeticNativeScalar)
    assert carrier.arithmetic.combine3(1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 0.0) == 20.0


def test_native_facade_selects_list_carrier() -> None:
    carrier = CarrierNative([1.0, 2.0, 3.0])

    assert isinstance(carrier.storage, CarrierStorageNativeList)
    assert isinstance(carrier.arithmetic, CarrierArithmeticNativeList)
    assert carrier.arithmetic.combine2(
        2.0,
        [1.0, 2.0, 3.0],
        3.0,
        [4.0, 5.0, 6.0],
        [0.0, 0.0, 0.0],
    ) == [14.0, 19.0, 24.0]


def test_native_facade_selects_tuple_carrier() -> None:
    carrier = CarrierNative((1.0, 2.0, 3.0))

    assert isinstance(carrier.storage, CarrierStorageNativeTuple)
    assert isinstance(carrier.arithmetic, CarrierArithmeticNativeTuple)
    assert carrier.arithmetic.combine2(
        2.0,
        (1.0, 2.0, 3.0),
        3.0,
        (4.0, 5.0, 6.0),
        (0.0, 0.0, 0.0),
    ) == (14.0, 19.0, 24.0)


def test_native_arithmetic_hot_path_has_no_type_branch() -> None:
    classes = (
        CarrierArithmeticNativeScalar,
        CarrierArithmeticNativeList,
        CarrierArithmeticNativeTuple,
    )

    for cls in classes:
        source = inspect.getsource(cls)
        assert "isinstance" not in source
