from stark.engines.jax.carriers.allocation import CarrierAllocationJax
from stark.engines.jax.carriers.basis import CarrierBasisJax
from stark.engines.jax.carriers.arithmetic import CarrierArithmeticJax
from stark.engines.jax.carriers.norm import CarrierNormJaxRMS
from stark.engines.jax.carriers.storage import CarrierJaxValue, CarrierStorageJax
from stark.engines.jax.carriers.validation import CarrierValidationJax


class CarrierJax:
    def __init__(self, template: CarrierJaxValue) -> None:
        storage = CarrierStorageJax.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationJax(storage)
        self.allocation = CarrierAllocationJax(storage)
        self.basis = CarrierBasisJax(storage)
        self.arithmetic = CarrierArithmeticJax()
        self.norm = CarrierNormJaxRMS()