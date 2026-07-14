from stark.engines.carrier_jax.allocation import CarrierAllocationJax
from stark.engines.carrier_jax.basis import CarrierBasisJax
from stark.engines.carrier_jax.arithmetic import CarrierArithmeticJax
from stark.engines.carrier_jax.norm import CarrierNormJaxRMS
from stark.engines.carrier_jax.storage import CarrierJaxValue, CarrierStorageJax
from stark.engines.carrier_jax.validation import CarrierValidationJax
from stark.engines.carriers import CarrierScalarItem


class CarrierJax:
    def __init__(self, template: CarrierJaxValue) -> None:
        storage = CarrierStorageJax.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationJax(storage)
        self.allocation = CarrierAllocationJax(storage)
        self.basis = CarrierBasisJax(storage)
        self.arithmetic = CarrierArithmeticJax()
        self.norm = CarrierNormJaxRMS()
        self.scalar = CarrierScalarItem()
