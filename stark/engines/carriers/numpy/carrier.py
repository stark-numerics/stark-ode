from stark.engines.carriers.numpy.allocation import CarrierAllocationNumpy
from stark.engines.carriers.numpy.basis import CarrierBasisNumpy
from stark.engines.carriers.numpy.arithmetic import CarrierArithmeticNumpy
from stark.engines.carriers.numpy.norm import CarrierNormNumpyRMS
from stark.engines.carriers.numpy.storage import CarrierNumpyValue, CarrierStorageNumpy
from stark.engines.carriers.numpy.validation import CarrierValidationNumpy


class CarrierNumpy:
    def __init__(self, template: CarrierNumpyValue) -> None:
        storage = CarrierStorageNumpy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationNumpy(storage)
        self.allocation = CarrierAllocationNumpy(storage)
        self.basis = CarrierBasisNumpy(storage)
        self.arithmetic = CarrierArithmeticNumpy(storage)
        self.norm = CarrierNormNumpyRMS()
