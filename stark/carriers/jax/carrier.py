from stark.carriers.jax.allocation import CarrierAllocationJax
from stark.carriers.jax.basis import CarrierBasisJax
from stark.carriers.jax.arithmetic import CarrierArithmeticJax
from stark.carriers.jax.norm import CarrierNormJaxRMS
from stark.carriers.jax.storage import CarrierJaxValue, CarrierStorageJax
from stark.carriers.jax.validation import CarrierValidationJax


class CarrierJax:
    def __init__(self, template: CarrierJaxValue) -> None:
        storage = CarrierStorageJax.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationJax(storage)
        self.allocation = CarrierAllocationJax(storage)
        self.basis = CarrierBasisJax(storage)
        self.arithmetic = CarrierArithmeticJax()
        self.norm = CarrierNormJaxRMS()