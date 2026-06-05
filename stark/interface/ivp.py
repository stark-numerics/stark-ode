from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from stark.carriers import CarrierNative, CarrierNumpy
from stark.contracts import Carrier
from stark.core import Configuration
from stark.integrator.integrator import Integrator
from stark.integrator.stepper import IntegratorStepper
from stark.interface.derivative import StarkDerivative
from stark.interface.vector import StarkVector, StarkVectorTranslation, StarkVectorAllocator
from stark.schemes import SchemeCashKarp


class IntervalLike(Protocol):
    present: Any
    step: Any
    stop: Any

    def copy(self) -> "IntervalLike": ...
    def increment(self) -> None: ...


DerivativeRuntime = Callable[[IntervalLike, StarkVector, StarkVectorTranslation], None]
SchemeFactory = Callable[[DerivativeRuntime, StarkVectorAllocator], Any]


@dataclass(slots=True)
class StarkIVPBuild:
    allocator: StarkVectorAllocator
    derivative: DerivativeRuntime
    scheme: Any
    configuration: Configuration
    stepper: IntegratorStepper
    integrator: Integrator
    initial: StarkVector
    interval: IntervalLike


@dataclass
class StarkIVP:
    derivative: Callable[..., Any] | StarkDerivative
    initial: Any
    interval: IntervalLike
    carrier: Carrier[Any, Any] | None = None
    scheme: SchemeFactory | Any | None = None
    configuration: Configuration | None = None

    prepared_initial: StarkVector = field(init=False)
    prepared_carrier: Carrier[Any, Any] = field(init=False)
    prepared_derivative: DerivativeRuntime = field(init=False)

    def __post_init__(self) -> None:
        self.interval = self.validate_interval(self.interval)
        self.prepared_initial = self.prepare_initial(self.initial)
        self.prepared_derivative = self.prepare_derivative(self.derivative)

    def validate_interval(self, interval: IntervalLike) -> IntervalLike:
        if isinstance(interval, (tuple, list)):
            raise TypeError(
                "StarkIVP requires an explicit interval-like object. "
                "Tuple/list intervals are not accepted."
            )

        required = ("present", "step", "stop")
        missing = [name for name in required if not hasattr(interval, name)]
        if missing:
            names = ", ".join(missing)
            raise TypeError(
                "StarkIVP interval must provide present, step, and stop attributes. "
                f"Missing: {names}."
            )

        methods = ("copy", "increment")
        missing_methods = [
            name for name in methods if not callable(getattr(interval, name, None))
        ]
        if missing_methods:
            names = ", ".join(missing_methods)
            raise TypeError(
                "StarkIVP interval must satisfy the IntervalLike contract. "
                f"Missing callable method(s): {names}."
            )

        for name in required:
            if getattr(interval, name) is None:
                raise ValueError(f"StarkIVP interval attribute {name!r} must not be None.")

        return interval

    def default_carrier_for(self, initial: Any) -> Carrier[Any, Any]:
        if self.is_numpy_array(initial):
            return CarrierNumpy(initial)

        return CarrierNative(initial)

    @staticmethod
    def is_numpy_array(value: Any) -> bool:
        try:
            import numpy as np
        except ImportError:
            return False

        return isinstance(value, np.ndarray)

    def prepare_initial(self, initial: Any) -> StarkVector:
        if isinstance(initial, StarkVector):
            if self.carrier is not None:
                raise TypeError(
                    "Cannot provide an explicit carrier when initial is already a StarkVector."
                )

            self.prepared_carrier = initial.carrier
            return initial

        carrier = self.carrier if self.carrier is not None else self.default_carrier_for(initial)
        value = carrier.validation.validate_state(initial)

        self.prepared_carrier = carrier
        return StarkVector(carrier.allocation.copy_state(value), carrier)

    def prepare_derivative(
        self,
        derivative: Callable[..., Any] | StarkDerivative,
    ) -> DerivativeRuntime:
        if isinstance(derivative, StarkDerivative):
            stark_derivative = derivative
        else:
            stark_derivative = StarkDerivative.from_callable(derivative)

        return stark_derivative.bind(self.prepared_carrier)

    def build(self) -> StarkIVPBuild:
        allocator = StarkVectorAllocator(self.prepared_carrier)
        derivative = self.prepared_derivative

        if self.scheme is None:
            scheme = SchemeCashKarp(derivative, allocator, configuration=self.configuration)
        elif isinstance(self.scheme, type):
            scheme = self.scheme(derivative, allocator, configuration=self.configuration)
        else:
            scheme = self.scheme

        configuration = self.configuration or Configuration()
        stepper = IntegratorStepper(scheme)
        integrator = Integrator(configuration=configuration)

        return StarkIVPBuild(
            allocator=allocator,
            derivative=derivative,
            scheme=scheme,
            configuration=configuration,
            stepper=stepper,
            integrator=integrator,
            initial=self.prepared_initial,
            interval=self.interval,
        )

    def integrate(self) -> Any:
        build = self.build()
        return build.integrator(
            build.stepper,
            build.interval,
            build.initial,
        )
