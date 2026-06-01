from stark.carriers.numpy.allocation import CarrierAllocationNumpy
from stark.carriers.numpy.basis import CarrierBasisNumpy
from stark.carriers.numpy.arithmetic import CarrierArithmeticNumpy
from stark.carriers.numpy.norm import CarrierNormNumpyRMS
from stark.carriers.numpy.storage import CarrierNumpyValue, CarrierStorageNumpy
from stark.carriers.numpy.validation import CarrierValidationNumpy


class CarrierNumpy:
    def __init__(self, template: CarrierNumpyValue) -> None:
        storage = CarrierStorageNumpy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationNumpy(storage)
        self.allocation = CarrierAllocationNumpy(storage)
        self.basis = CarrierBasisNumpy(storage)
        self.arithmetic = CarrierArithmeticNumpy(storage)
        self.norm = CarrierNormNumpyRMS()
