from typing import Any, Protocol

from stark.carriers.core import CarrierBound


class Convention(Protocol):
    def __call__(
        self,
        function: Any,
        t: Any,
        y: Any,
        dy: Any,
        carrier: CarrierBound,
    ) -> Any:
        ...


class ConventionReturn:
    def __call__(
        self,
        function: Any,
        t: Any,
        y: Any,
        dy: Any,
        carrier: CarrierBound,
    ) -> Any:
        value = function(t, y)
        return carrier.coerce_translation(value)


class ConventionInPlace:
    def __call__(
        self,
        function: Any,
        t: Any,
        y: Any,
        dy: Any,
        carrier: CarrierBound,
    ) -> Any:
        function(t, y, dy)
        carrier.validate_translation(dy)
        return dy