from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from stark.contracts import IntervalLike, State
from stark.contracts.engine import StarkEngine
from stark.core.configuration import Configuration
from stark.integrator.integrator import Checkpoints, Integrator
from stark.integrator.stepper import IntegratorStepper
from stark.interface.derivative import StarkDerivative, StarkDerivativeSignature
from stark.interface.layout import StarkLayout
from stark.interface.method import StarkMethod, StarkMethodError


EngineFactory = Callable[[StarkLayout], StarkEngine]


@dataclass(slots=True)
class StarkSystemIVP:
    """
    Prepared initial-value problem built from a `StarkSystem`.

    The IVP owns the engine state, interval, scheme, stepper, and integrator
    created from a system/method/engine recipe. It is reusable: each
    `stable_trajectory(...)` or `mutating_trajectory(...)` call starts from a
    fresh copy of the prepared initial state and interval unless explicit
    working objects are supplied.

    Use `stable_trajectory(...)` when you want to collect or compare output.
    It yields copied interval/state snapshots, so previously yielded objects do
    not change as integration continues. Use `mutating_trajectory(...)` for
    benchmarks, repeated solves, and low-overhead observation. It yields the
    mutable working interval/state objects themselves, so each yielded pair is
    the same objects at a later point in the solve.
    """

    system: StarkSystem
    method: StarkMethod
    engine: StarkEngine
    initial: State
    interval: IntervalLike
    scheme: object
    stepper: IntegratorStepper
    integrator: Integrator
    configuration: Configuration

    def fresh_state(self) -> State:
        """Return a fresh mutable state copied from the prepared initial state."""

        state = self.engine.allocator.allocate_state()
        self.engine.allocator.copy_state(self.initial, state)
        return state

    def fresh_interval(self) -> IntervalLike:
        """Return a fresh interval copied from the prepared initial interval."""

        return self.interval.copy()

    def stable_trajectory(
        self,
        *,
        interval: IntervalLike | None = None,
        state: State | None = None,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        """
        Yield stable copied snapshots from a solve.

        This is the safe trajectory form for collected output: each yielded
        interval and state is a snapshot that will not be mutated by later
        steps. By default the solve starts from fresh copies of the prepared
        interval and initial state.
        """

        working_interval = self.fresh_interval() if interval is None else interval
        working_state = self.fresh_state() if state is None else state
        return self.integrator(self.stepper, working_interval, working_state, checkpoints)

    def mutating_trajectory(
        self,
        *,
        interval: IntervalLike | None = None,
        state: State | None = None,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        """
        Yield the mutable working interval and state from a solve.

        This is the low-overhead trajectory form for benchmarks, repeated
        solves, and streaming observation. Every yielded pair contains the same
        interval and state objects as they mutate through the integration. By
        default the solve starts from fresh copies of the prepared interval and
        initial state.
        """

        working_interval = self.fresh_interval() if interval is None else interval
        working_state = self.fresh_state() if state is None else state
        return self.integrator.mutating_trajectory(
            self.stepper,
            working_interval,
            working_state,
            checkpoints,
        )

    def integrate(self, checkpoints: Checkpoints | None = None) -> Iterator[tuple[IntervalLike, State]]:
        return self.stable_trajectory(checkpoints=checkpoints)


@dataclass(frozen=True, slots=True)
class StarkSystem:
    """
    User-facing declaration of an ODE system over a `StarkLayout`.

    A system combines the derivative with the layout that names the state
    fields, translation fields, shapes, and norm policy. Calling `ivp(...)`
    supplies the remaining runtime choices: initial values, an interval, a
    `StarkMethod`, an engine class or factory, and optional configuration. The
    system then asks the engine for backend storage/algebra support, prepares
    the derivative for that accelerator, constructs the method stack, and
    returns a reusable `StarkSystemIVP`.

    `linearizer` and `inner_product` are optional problem-level ingredients used
    by implicit method stacks when the selected resolvent asks for them.
    """

    derivative: object
    layout: StarkLayout
    linearizer: object | None = None
    inner_product: object | None = None

    def ivp(
        self,
        *,
        initial: object,
        interval: IntervalLike,
        method: StarkMethod,
        engine: EngineFactory,
        configuration: Configuration | None = None,
    ) -> StarkSystemIVP:
        configuration = configuration if configuration is not None else Configuration()
        prepared_engine = engine(self.layout)
        prepared_initial = self.prepare_initial(initial, prepared_engine)
        prepared_derivative = self.prepare_derivative(prepared_engine)
        scheme = self.prepare_scheme(method, prepared_engine, prepared_derivative, configuration)
        stepper = IntegratorStepper(scheme)
        integrator = Integrator(configuration=configuration)

        return StarkSystemIVP(
            system=self,
            method=method,
            engine=prepared_engine,
            initial=prepared_initial,
            interval=interval,
            scheme=scheme,
            stepper=stepper,
            integrator=integrator,
            configuration=configuration,
        )

    def prepare_initial(self, initial: object, engine: StarkEngine) -> object:
        state = engine.allocator.allocate_state()
        for field, carrier in zip(engine.algebraist_layout.fields, engine.carriers, strict=True):
            value = self.initial_value(initial, field.state_path)
            validated = carrier.validation.validate_state(value)
            field.state_path.set(state, carrier.allocation.copy_state(validated))
        return state

    def initial_value(self, initial: object, path: object) -> object:
        value = initial
        parts = getattr(path, "parts", None)
        if parts is None:
            raise TypeError("StarkSystem initial paths must expose path parts.")
        for part in parts:
            if isinstance(value, Mapping):
                try:
                    value = value[part]
                except KeyError as exc:
                    raise KeyError(f"Initial values are missing field path {path!s}.") from exc
            else:
                try:
                    value = getattr(value, part)
                except AttributeError as exc:
                    raise AttributeError(f"Initial value object is missing field path {path!s}.") from exc
        return value

    def prepare_scheme(
        self,
        method: StarkMethod,
        engine: StarkEngine,
        derivative: object,
        configuration: Configuration,
    ) -> object:
        inverter = None
        if method.inverter is not None:
            inverter = self.construct_component(
                "inverter",
                method.inverter,
                available={
                    "allocator": engine.allocator,
                    "configuration": configuration,
                    "accelerator": engine.accelerator,
                },
                options=method.inverter_options,
            )

        resolvent = None
        if method.resolvent is not None:
            resolvent = self.construct_component(
                "resolvent",
                method.resolvent,
                available={
                    "allocator": engine.allocator,
                    "configuration": configuration,
                    "accelerator": engine.accelerator,
                    "inverter": inverter,
                    "linearizer": self.linearizer,
                    "inner_product": self.inner_product,
                },
                options=method.resolvent_options,
            )

        return self.construct_component(
            "scheme",
            method.scheme,
            available={
                "derivative": derivative,
                "allocator": engine.allocator,
                "configuration": configuration,
                "specialist": engine.algebraist_specialist,
                "resolvent": resolvent,
            },
            options=method.scheme_options,
        )

    def prepare_derivative(self, engine: StarkEngine) -> object:
        derivative = self.derivative
        if isinstance(derivative, StarkDerivative):
            return derivative.accelerate(engine.accelerator)
        if isinstance(derivative, StarkDerivativeSignature):
            return StarkDerivative(derivative).accelerate(engine.accelerator)
        if callable(derivative):
            return StarkDerivative(derivative).accelerate(engine.accelerator)
        raise TypeError("StarkSystem derivative must be callable or a derivative signature.")

    def construct_component(
        self,
        role: str,
        component: type[Any],
        *,
        available: Mapping[str, object | None],
        options: Mapping[str, object],
    ) -> object:
        parameters = signature(component).parameters
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        unsupported = tuple(
            name
            for name in options
            if name not in parameters and not accepts_kwargs
        )
        if unsupported:
            names = ", ".join(unsupported)
            raise StarkMethodError(
                f"{component.__name__} does not accept {role} option(s): {names}."
            )

        kwargs = {
            name: value
            for name, value in available.items()
            if value is not None and (name in parameters or accepts_kwargs)
        }
        kwargs.update(options)

        try:
            return component(**kwargs)
        except TypeError as exc:
            raise StarkMethodError(
                f"Could not construct {role} {component.__name__}: {exc}"
            ) from exc


__all__ = ["StarkSystem", "StarkSystemIVP"]
