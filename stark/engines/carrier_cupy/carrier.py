from stark.engines.carrier_cupy.allocation import CarrierAllocationCupy
from stark.engines.carrier_cupy.basis import CarrierBasisCupy
from stark.engines.carrier_cupy.arithmetic import CarrierArithmeticCupy
from stark.engines.carrier_cupy.norm import CarrierNormCupyRMS
from stark.engines.carrier_cupy.storage import CarrierCupyValue, CarrierStorageCupy
from stark.engines.carrier_cupy.validation import CarrierValidationCupy
from stark.engines.carriers import CarrierScalarItem


class CarrierCupy:
    def __init__(self, template: CarrierCupyValue) -> None:
        storage = CarrierStorageCupy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationCupy(storage)
        self.allocation = CarrierAllocationCupy(storage)
        self.basis = CarrierBasisCupy(storage)
        self.arithmetic = CarrierArithmeticCupy(storage)
        self.norm = CarrierNormCupyRMS()
        self.scalar = CarrierScalarItem()
