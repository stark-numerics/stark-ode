from dataclasses import dataclass
from typing import Generic, Literal, Protocol

from stark.contracts.translations import (
    StateTypeContravariant, 
    TranslationTypeContravariant, 
    StateType, 
    TranslationType
)

class CarrierStorage(Protocol[StateTypeContravariant, TranslationTypeContravariant]):
    def is_state(self, value: StateTypeContravariant) -> bool: ...
    def is_translation(self, value: TranslationTypeContravariant) -> bool: ...

class CarrierValidation(Protocol[StateType, TranslationType]):
    def validate_state(self, value: StateType) -> StateType: ...
    def validate_translation(self, value: TranslationType) -> TranslationType: ...
    def coerce_translation(self, value: object) -> TranslationType: ...

class CarrierAllocation(Protocol[StateType, TranslationType]):
    def zero_state(self) -> StateType: ...
    def zero_translation(self) -> TranslationType: ...
    def copy_state(self, value: StateType) -> StateType: ...
    def copy_translation(self, value: TranslationType) -> TranslationType: ...
    def allocate_translation(self) -> TranslationType: ...

class CarrierArithmetic(Protocol[StateType, TranslationType]):
    preference: Literal["return", "into"]

    def translate(
        self,
        state: StateType,
        step: float,
        derivative: TranslationType,
        result: StateType,
    ) -> StateType | None: ...

    def add(
        self,
        left: TranslationType,
        right: TranslationType,
        result: TranslationType,
    ) -> TranslationType | None: ...

    def scale(
        self,
        factor: float,
        value: TranslationType,
        result: TranslationType,
    ) -> TranslationType | None: ...

    def combine(
        self,
        terms: tuple[tuple[float, TranslationType], ...],
        result: TranslationType,
    ) -> TranslationType | None: ...

class CarrierNorm(Protocol[TranslationTypeContravariant]):
    def __call__(self, value: TranslationTypeContravariant) -> float: ...

@dataclass(frozen=True)
class Carrier(Generic[StateType, TranslationType]):
    storage: CarrierStorage
    validation: CarrierValidation[StateType, TranslationType]
    allocation: CarrierAllocation[StateType, TranslationType]
    arithmetic: CarrierArithmetic[StateType, TranslationType]
    norm: CarrierNorm
    