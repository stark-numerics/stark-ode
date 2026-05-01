from dataclasses import dataclass, field
from typing import Any

from stark import Executor, Integrator, Marcher
from stark.carriers import CarrierError, CarrierLibrary
from stark.interface.derivative import StarkDerivative
from stark.interface.vector import StarkVector, StarkVectorWorkbench
from stark.routing import Routing, RoutingVector
from stark.schemes import SchemeCashKarp


@dataclass(slots=True)
class StarkIVPBuild:
    workbench: StarkVectorWorkbench
    derivative: Any
    scheme: Any
    executor: Executor
    marcher: Marcher
    integrator: Integrator
    initial: StarkVector
    interval: Any


@dataclass
class StarkIVP:
    derivative: Any
    initial: Any
    interval: Any
    carrier: Any = None
    carrier_library: CarrierLibrary | None = None
    routing: Routing | None = None
    scheme: Any = None
    executor: Any = None

    prepared_initial: StarkVector = field(init=False)
    prepared_carrier: Any = field(init=False)
    vector_routing: RoutingVector = field(init=False)
    prepared_derivative: Any = field(init=False)

    def __post_init__(self) -> None:
        self.interval = self.validate_interval(self.interval)

        if self.carrier is None and self.carrier_library is None:
            self.carrier_library = CarrierLibrary.default()

        self.prepared_initial = self.prepare_initial(self.initial)
        self.prepared_derivative = self.prepare_derivative(self.derivative)

    def validate_interval(self, interval: Any) -> Any:
        if isinstance(interval, (tuple, list)):
            raise TypeError(
                "StarkIVP requires an explicit interval-like object. Tuple/list "
                "intervals are not accepted."
            )

        required = ("present", "step", "stop")

        missing = [
            name
            for name in required
            if not hasattr(interval, name)
        ]

        if missing:
            names = ", ".join(missing)
            raise TypeError(
                "StarkIVP interval must provide present, step, and stop attributes. "
                f"Missing: {names}."
            )

        methods = ("copy", "increment")
        missing_methods = [
            name
            for name in methods
            if not callable(getattr(interval, name, None))
        ]

        if missing_methods:
            names = ", ".join(missing_methods)
            raise TypeError(
                "StarkIVP interval must satisfy the IntervalLike contract. "
                f"Missing callable method(s): {names}."
            )

        for name in required:
            if getattr(interval, name) is None:
                raise ValueError(
                    f"StarkIVP interval attribute {name!r} must not be None."
                )

        return interval

    def resolve_vector_routing(self, carrier: Any) -> RoutingVector:
        if self.routing is None:
            return carrier.recommend_vector_routing()

        if not isinstance(self.routing, Routing):
            raise TypeError(
                "StarkIVP routing must be a Routing object. "
                "Pass Routing(vector=...) to override vector routing."
            )

        return self.routing.policy(RoutingVector)

    def prepare_initial(self, initial: Any) -> StarkVector:
        if isinstance(initial, StarkVector):
            if self.carrier is not None:
                raise CarrierError(
                    "Cannot provide an explicit carrier when initial is already a "
                    "StarkVector."
                )

            self.prepared_carrier = initial.carrier
            self.vector_routing = self.resolve_vector_routing(initial.carrier.carrier)
            return initial

        carrier = self.carrier

        if carrier is None:
            carrier_library = self.carrier_library or CarrierLibrary.default()
            self.carrier_library = carrier_library
            carrier = carrier_library.carrier_for(initial)

        value = carrier.coerce_state(initial)
        bound_carrier = carrier.bind(value)

        self.prepared_carrier = bound_carrier
        self.vector_routing = self.resolve_vector_routing(carrier)

        return StarkVector(value, bound_carrier)

    def prepare_derivative(self, derivative: Any) -> Any:
        if isinstance(derivative, StarkDerivative):
            stark_derivative = derivative
        else:
            stark_derivative = StarkDerivative.from_callable(derivative)

        return stark_derivative.bind(self.prepared_carrier)

    def build(self) -> StarkIVPBuild:
        workbench = StarkVectorWorkbench(
            self.prepared_carrier,
            self.vector_routing,
        )

        derivative = self.prepared_derivative

        if self.scheme is None:
            scheme = SchemeCashKarp(derivative, workbench)
        elif isinstance(self.scheme, type):
            scheme = self.scheme(derivative, workbench)
        else:
            scheme = self.scheme

        executor = self.executor or Executor()
        marcher = Marcher(scheme, executor)
        integrator = Integrator(executor=executor)

        return StarkIVPBuild(
            workbench=workbench,
            derivative=derivative,
            scheme=scheme,
            executor=executor,
            marcher=marcher,
            integrator=integrator,
            initial=self.prepared_initial,
            interval=self.interval,
        )

    def integrate(self) -> Any:
        build = self.build()

        return build.integrator(
            build.marcher,
            build.interval,
            build.initial,
        )
