from stark.engines.carriers.cupy.allocation import CarrierAllocationCupy
from stark.engines.carriers.cupy.basis import CarrierBasisCupy
from stark.engines.carriers.cupy.arithmetic import CarrierArithmeticCupy
from stark.engines.carriers.cupy.norm import CarrierNormCupyRMS
from stark.engines.carriers.cupy.storage import CarrierCupyValue, CarrierStorageCupy
from stark.engines.carriers.cupy.validation import CarrierValidationCupy


class CarrierCupy:
    def __init__(self, template: CarrierCupyValue) -> None:
        storage = CarrierStorageCupy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationCupy(storage)
        self.allocation = CarrierAllocationCupy(storage)
        self.basis = CarrierBasisCupy(storage)
        self.arithmetic = CarrierArithmeticCupy(storage)
        self.norm = CarrierNormCupyRMS()