def test_carrier_package_exports_minimal_public_objects():
    from stark.carriers import DeprecatedCarrierLibrary, DeprecatedCarrierNative, DeprecatedCarrierNumpy

    assert DeprecatedCarrierLibrary is not None
    assert DeprecatedCarrierNative is not None
    assert DeprecatedCarrierNumpy is not None


def test_carrier_package_exports_kernel_and_norm_objects():
    from stark.carriers import (
        DeprecatedCarrierKernel,
        DeprecatedCarrierKernelNative,
        DeprecatedCarrierKernelNumpy,
        DeprecatedCarrierNorm,
        DeprecatedCarrierNormNativeRMS,
        DeprecatedCarrierNormNumpyRMS,
        DeprecatedCarrierNormNumpyMax,
    )

    assert DeprecatedCarrierKernel is not None
    assert DeprecatedCarrierKernelNative is not None
    assert DeprecatedCarrierKernelNumpy is not None
    assert DeprecatedCarrierNorm is not None
    assert DeprecatedCarrierNormNativeRMS is not None
    assert DeprecatedCarrierNormNumpyRMS is not None
    assert DeprecatedCarrierNormNumpyMax is not None


def test_carrier_objects_are_not_root_exported_yet():
    import stark

    assert not hasattr(stark, "DeprecatedCarrierLibrary")
    assert not hasattr(stark, "DeprecatedCarrierNative")
    assert not hasattr(stark, "DeprecatedCarrierNumpy")

def test_carrier_error_is_exported():
    from stark.carriers import DeprecatedCarrierError

    assert issubclass(DeprecatedCarrierError, ValueError)