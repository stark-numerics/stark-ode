from stark.carriers.cupy.allocation import CarrierAllocationCupy
from stark.carriers.cupy.arithmetic import CarrierArithmeticCupy
from stark.carriers.cupy.norm import CarrierNormCupyRMS
from stark.carriers.cupy.storage import CarrierCupyValue, CarrierStorageCupy
from stark.carriers.cupy.validation import CarrierValidationCupy


class CarrierCupy:
    def __init__(self, template: CarrierCupyValue) -> None:
        storage = CarrierStorageCupy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationCupy(storage)
        self.allocation = CarrierAllocationCupy(storage)
        self.arithmetic = CarrierArithmeticCupy(storage)
        self.norm = CarrierNormCupyRMS()