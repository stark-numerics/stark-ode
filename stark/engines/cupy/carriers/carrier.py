from stark.engines.cupy.carriers.allocation import CarrierAllocationCupy
from stark.engines.cupy.carriers.basis import CarrierBasisCupy
from stark.engines.cupy.carriers.arithmetic import CarrierArithmeticCupy
from stark.engines.cupy.carriers.norm import CarrierNormCupyRMS
from stark.engines.cupy.carriers.storage import CarrierCupyValue, CarrierStorageCupy
from stark.engines.cupy.carriers.validation import CarrierValidationCupy


class CarrierCupy:
    def __init__(self, template: CarrierCupyValue) -> None:
        storage = CarrierStorageCupy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationCupy(storage)
        self.allocation = CarrierAllocationCupy(storage)
        self.basis = CarrierBasisCupy(storage)
        self.arithmetic = CarrierArithmeticCupy(storage)
        self.norm = CarrierNormCupyRMS()