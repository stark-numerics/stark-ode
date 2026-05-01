def test_carrier_package_exports_minimal_public_objects():
    from stark.carriers import CarrierLibrary, CarrierNative, CarrierNumpy

    assert CarrierLibrary is not None
    assert CarrierNative is not None
    assert CarrierNumpy is not None


def test_carrier_package_exports_kernel_and_norm_objects():
    from stark.carriers import (
        CarrierKernel,
        CarrierKernelNative,
        CarrierKernelNumpy,
        CarrierNorm,
        CarrierNormNativeRMS,
        CarrierNormNumpyRMS,
        CarrierNormNumpyMax,
    )

    assert CarrierKernel is not None
    assert CarrierKernelNative is not None
    assert CarrierKernelNumpy is not None
    assert CarrierNorm is not None
    assert CarrierNormNativeRMS is not None
    assert CarrierNormNumpyRMS is not None
    assert CarrierNormNumpyMax is not None


def test_carrier_objects_are_not_root_exported_yet():
    import stark

    assert not hasattr(stark, "CarrierLibrary")
    assert not hasattr(stark, "CarrierNative")
    assert not hasattr(stark, "CarrierNumpy")

def test_carrier_error_is_exported():
    from stark.carriers import CarrierError

    assert issubclass(CarrierError, ValueError)