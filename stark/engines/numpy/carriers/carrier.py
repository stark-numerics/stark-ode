from stark.engines.numpy.carriers.allocation import CarrierAllocationNumpy
from stark.engines.numpy.carriers.basis import CarrierBasisNumpy
from stark.engines.numpy.carriers.arithmetic import CarrierArithmeticNumpy
from stark.engines.numpy.carriers.norm import CarrierNormNumpyRMS
from stark.engines.numpy.carriers.storage import CarrierNumpyValue, CarrierStorageNumpy
from stark.engines.numpy.carriers.validation import CarrierValidationNumpy


class CarrierNumpy:
    def __init__(self, template: CarrierNumpyValue) -> None:
        storage = CarrierStorageNumpy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationNumpy(storage)
        self.allocation = CarrierAllocationNumpy(storage)
        self.basis = CarrierBasisNumpy(storage)
        self.arithmetic = CarrierArithmeticNumpy(storage)
        self.norm = CarrierNormNumpyRMS()
