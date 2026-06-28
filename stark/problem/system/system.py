"""Problem-layer assembly for reusable initial-value problems.

`System` is the user-facing point where a derivative, frame, optional
linearizer, method, engine, interval, and initial state become a prepared
`SystemIVP`. The resulting IVP keeps the expensive preparation work together
so scripts can run final-state solves, trajectories, benchmarks, and
comparisons without rebuilding the whole stack each time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from stark.core.contracts import DerivativeSplitLike, IntervalLike, State
from stark.core.contracts.engine import Engine
from stark.core.configuration import Configuration
from stark.core.integrator.integrator import Checkpoints, Integrator
from stark.core.integrator.stepper import IntegratorStepper
from stark.problem.derivative import Derivative, DerivativeSignature
from stark.problem.derivative.split import DerivativeSplit
from stark.problem.frame.frame import Frame
from stark.problem.linearizer.linearizer import Linearizer
from stark.problem.linearizer.implementation import LinearizerImplementation
from stark.problem.linearizer.signature import LinearizerSignature
from stark.methods.method import Method, MethodError


EngineFactory = Callable[[Frame], Engine]


@dataclass(frozen=True, slots=True)
class SystemFinalResult:
    """
    Final working interval, state, and step count from a prepared IVP solve.

    `SystemIVP.final_result(...)` returns this object when the caller
    wants the destination rather than the trajectory. The `interval` and
    `state` are the mutable working objects used by the solve, positioned at
    the final accepted state. `steps` is the number of accepted integration
    steps taken to reach that state.
    """

    interval: IntervalLike
    state: State
    steps: int


@dataclass(slots=True)
class SystemIVP:
    """
    Prepared initial-value problem built from a `System`.

    The IVP owns the concrete runtime objects created from a
    system/method/engine recipe: backend engine, prepared initial state,
    interval template, scheme, stepper, integrator, and configuration. It is
    reusable. Helper methods start from fresh copies of the prepared initial
    state and interval unless explicit working objects are supplied.

    Choose the access pattern by the kind of output you need:

    - `final_result(...)` integrates to the end and returns the final working
      interval/state plus the accepted step count.
    - `stable_trajectory(...)` yields copied interval/state snapshots, suitable
      for collecting output without later mutation changing earlier samples.
    - `mutating_trajectory(...)` yields the mutable working interval/state
      objects themselves, suitable for benchmarks and streaming observation.
    - `integrate(...)` is a short alias for the stable trajectory path.
    """

    system: System
    method: Method
    engine: Engine
    initial: State
    interval: IntervalLike
    scheme: object
    stepper: IntegratorStepper
    integrator: Integrator
    configuration: Configuration

    def fresh_state(self) -> State:
        """
        Return a fresh mutable state copied from the prepared initial state.

        Use this when you want to manage the working state yourself while still
        reusing the engine-owned storage shape prepared by the IVP.
        """

        state = self.engine.allocator.allocate_state()
        self.engine.allocator.copy_state(self.initial, state)
        return state

    def fresh_interval(self) -> IntervalLike:
        """
        Return a fresh interval copied from the prepared interval template.

        This is useful for repeated solves where the time interval should start
        from the same present/step/stop values each time.
        """

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
        interval and initial state. Pass explicit `interval` or `state` objects
        to continue from caller-owned working objects.
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
        initial state. Pass explicit `interval` or `state` objects to continue
        from caller-owned working objects.
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
        """
        Yield stable copied trajectory snapshots.

        This is a concise alias for `stable_trajectory(checkpoints=...)` kept
        for examples and simple scripts where copied trajectory output is the
        natural default.
        """

        return self.stable_trajectory(checkpoints=checkpoints)

    def final_result(
        self,
        *,
        interval: IntervalLike | None = None,
        state: State | None = None,
    ) -> SystemFinalResult:
        """
        Integrate to the final time and return the final working objects.

        By default this starts from fresh copies of the prepared interval and
        initial state. The returned state and interval are the mutable working
        objects used by the solve, and `steps` counts accepted integration
        steps.
        """

        working_interval = self.fresh_interval() if interval is None else interval
        working_state = self.fresh_state() if state is None else state
        steps = 0
        for _interval, _state in self.mutating_trajectory(
            interval=working_interval,
            state=working_state,
        ):
            steps += 1
        return SystemFinalResult(
            interval=working_interval,
            state=working_state,
            steps=steps,
        )


@dataclass(frozen=True, slots=True)
class System:
    """
    User-facing declaration of an ODE system over a `Frame`.

    A system combines the derivative with the frame that names the state
    fields, translation fields, shapes, and norm policy. Calling `ivp(...)`
    supplies the remaining runtime choices: initial values, an interval, a
    `Method`, an engine class or factory, and optional configuration. The
    system then asks the engine for backend storage/algebra support, prepares
    the derivative for that accelerator, constructs the method stack, and
    returns a reusable `SystemIVP`.

    `linearizer` and `inner_product` are optional problem-level ingredients used
    by implicit method stacks when the selected resolvent asks for them.
    """

    derivative: object
    frame: Frame
    linearizer: object | None = None
    inner_product: object | None = None

    def ivp(
        self,
        *,
        initial: object,
        interval: IntervalLike,
        method: Method,
        engine: EngineFactory,
        configuration: Configuration | None = None,
    ) -> SystemIVP:
        """
        Build a reusable IVP from this system and a concrete runtime recipe.

        `initial` supplies values matching the system frame. The engine
        factory receives the frame and returns the backend bundle used by the
        scheme stack: allocator, carrier support, accelerator, algebraist
        frame, and algebraist kernels. `method` names the scheme and any
        resolvent/inverter components needed by that scheme.

        The returned `SystemIVP` keeps the prepared objects together. It
        can be reused for repeated solves via `final_result(...)`,
        `integrate(...)`, `stable_trajectory(...)`, or
        `mutating_trajectory(...)`.
        """

        configuration = configuration if configuration is not None else Configuration()
        prepared_engine = engine(self.frame)
        prepared_initial = self.prepare_initial(initial, prepared_engine)
        prepared_derivative = self.prepare_derivative(prepared_engine)
        scheme = self.prepare_scheme(method, prepared_engine, prepared_derivative, configuration)
        stepper = IntegratorStepper(scheme)
        integrator = Integrator(configuration=configuration)

        return SystemIVP(
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

    def prepare_initial(self, initial: object, engine: Engine) -> object:
        state = engine.allocator.allocate_state()
        for field, carrier in zip(engine.algebraist_frame.fields, engine.carriers, strict=True):
            value = self.initial_value(initial, field.state_path)
            validated = carrier.validation.validate_state(value)
            field.state_path.set(state, carrier.allocation.copy_state(validated))
        return state

    def initial_value(self, initial: object, path: object) -> object:
        value = initial
        parts = getattr(path, "parts", None)
        if parts is None:
            raise TypeError("System initial paths must expose path parts.")
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
        method: Method,
        engine: Engine,
        derivative: object,
        configuration: Configuration,
    ) -> object:
        prepared_linearizer = self.prepare_linearizer(engine)

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
                    "linearizer": prepared_linearizer,
                    "inner_product": self.inner_product
                    if self.inner_product is not None
                    else getattr(engine.allocator, "inner_product", None),
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

    def prepare_linearizer(self, engine: Engine) -> LinearizerImplementation | None:
        linearizer = self.linearizer
        if linearizer is None:
            return None
        if isinstance(linearizer, Linearizer):
            return linearizer.accelerate(engine.accelerator)
        if isinstance(linearizer, LinearizerSignature):
            return Linearizer(linearizer).accelerate(engine.accelerator)
        if callable(linearizer):
            return Linearizer(linearizer).accelerate(engine.accelerator)
        raise TypeError("System linearizer must be callable or a linearizer signature.")

    def prepare_derivative(self, engine: Engine) -> object:
        derivative = self.derivative
        if isinstance(derivative, DerivativeSplitLike):
            return DerivativeSplit(
                implicit=self.prepare_derivative_part(derivative.implicit, engine),
                explicit=self.prepare_derivative_part(derivative.explicit, engine),
            )
        return self.prepare_derivative_part(derivative, engine)

    def prepare_derivative_part(self, derivative: object, engine: Engine) -> object:
        if isinstance(derivative, Derivative):
            return derivative.accelerate(engine.accelerator)
        if isinstance(derivative, DerivativeSignature):
            return Derivative(derivative).accelerate(engine.accelerator)
        if callable(derivative):
            return Derivative(derivative).accelerate(engine.accelerator)
        raise TypeError("System derivative must be callable or a derivative signature.")

    def construct_component(
        self,
        role: str,
        component: object,
        *,
        available: Mapping[str, object | None],
        options: Mapping[str, object],
    ) -> object:
        if not isinstance(component, type):
            if options:
                raise MethodError(
                    f"Could not construct {role}: ready component instances do not accept options."
                )
            return component

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
            raise MethodError(
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
            raise MethodError(
                f"Could not construct {role} {component.__name__}: {exc}"
            ) from exc


__all__ = ["System", "SystemFinalResult", "SystemIVP"]
