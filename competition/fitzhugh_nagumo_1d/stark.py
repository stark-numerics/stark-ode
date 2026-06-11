from __future__ import annotations

from typing import Any

import numpy as np

from stark.comparison import Comparison
from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.engines.numpy.engine import EngineNumpy
from stark.interface.derivative import DerivativeStyle
from stark.interface.layout import Layout
from stark.methods.method import Method
from stark.interface.system import System
from stark.methods.resolvents.secant.anderson import ResolventAnderson
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_7 import SchemeKennedyCarpenter43_7
from stark.methods.schemes.implicit.adaptive.kvaerno3 import SchemeKvaerno3


Array = Any


@DerivativeStyle.kernel(state=("u", "v"), translation=("du", "dv"))
def fitzhugh_nagumo_rhs(
    u: Array,
    v: Array,
    du: Array,
    dv: Array,
    diffusivity_u: float,
    epsilon: float,
    a: float,
    b: float,
    inv_dx2: float,
) -> None:
    size = u.size
    for index in range(size):
        left = u[index - 1 if index > 0 else size - 1]
        centre = u[index]
        right = u[index + 1 if index + 1 < size else 0]
        laplacian_u = (left - 2.0 * centre + right) * inv_dx2
        du[index] = (
            diffusivity_u * laplacian_u
            + centre
            - (centre * centre * centre) / 3.0
            - v[index]
        )
        dv[index] = epsilon * (centre + a - b * v[index])


@DerivativeStyle.kernel(state=("u", "v"), translation=("du", "dv"))
def fitzhugh_nagumo_explicit_rhs(
    u: Array,
    v: Array,
    du: Array,
    dv: Array,
    epsilon: float,
    a: float,
    b: float,
) -> None:
    size = u.size
    for index in range(size):
        centre = u[index]
        du[index] = centre - (centre * centre * centre) / 3.0 - v[index]
        dv[index] = epsilon * (centre + a - b * v[index])


@DerivativeStyle.kernel(state=("u", "v"), translation=("du", "dv"))
def fitzhugh_nagumo_implicit_rhs(
    u: Array,
    _v: Array,
    du: Array,
    dv: Array,
    diffusivity_u: float,
    inv_dx2: float,
) -> None:
    size = u.size
    for index in range(size):
        left = u[index - 1 if index > 0 else size - 1]
        centre = u[index]
        right = u[index + 1 if index + 1 < size else 0]
        du[index] = diffusivity_u * (left - 2.0 * centre + right) * inv_dx2
        dv[index] = 0.0


class FitzHughNagumoParameters:
    __slots__ = (
        "a",
        "b",
        "diffusivity_u",
        "epsilon",
        "grid_size",
        "initial_step",
        "length",
        "resolution_atol",
        "resolution_max_iterations",
        "resolution_rtol",
        "t_start",
        "t_stop",
        "tolerance_atol",
        "tolerance_rtol",
    )

    def __init__(self, problem_parameters, stark_parameters) -> None:
        self.grid_size = int(problem_parameters["grid_size"])
        self.length = float(problem_parameters["length"])
        self.diffusivity_u = float(problem_parameters["diffusivity_u"])
        self.epsilon = float(problem_parameters["epsilon"])
        self.a = float(problem_parameters["a"])
        self.b = float(problem_parameters["b"])
        self.t_start = float(problem_parameters["t0"])
        self.t_stop = float(problem_parameters["t1"])
        self.initial_step = float(stark_parameters["step"])
        self.tolerance_atol = float(stark_parameters["tolerance_atol"])
        self.tolerance_rtol = float(stark_parameters["tolerance_rtol"])
        self.resolution_atol = float(stark_parameters["resolution_atol"])
        self.resolution_rtol = float(stark_parameters["resolution_rtol"])
        self.resolution_max_iterations = int(stark_parameters["resolution_max_iterations"])

    @property
    def dx(self) -> float:
        return self.length / self.grid_size

    @property
    def inv_dx2(self) -> float:
        return 1.0 / (self.dx * self.dx)


def full_derivative(parameters: FitzHughNagumoParameters):
    return fitzhugh_nagumo_rhs.with_parameters(
        parameters.diffusivity_u,
        parameters.epsilon,
        parameters.a,
        parameters.b,
        parameters.inv_dx2,
    )


def imex_derivative(parameters: FitzHughNagumoParameters):
    return DerivativeStyle.imex(
        implicit=fitzhugh_nagumo_implicit_rhs.with_parameters(
            parameters.diffusivity_u,
            parameters.inv_dx2,
        ),
        explicit=fitzhugh_nagumo_explicit_rhs.with_parameters(
            parameters.epsilon,
            parameters.a,
            parameters.b,
        ),
    )


class FitzHughNagumoSpectralResolvent:
    __slots__ = ("operator_symbol", "tableau", "u_hat")

    def __init__(self, parameters: FitzHughNagumoParameters, tableau=None) -> None:
        self.tableau = tableau
        grid_size = int(parameters.grid_size)
        theta = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=1.0)
        self.operator_symbol = (
            parameters.diffusivity_u
            * 2.0
            * (np.cos(theta) - 1.0)
            * parameters.inv_dx2
        )
        self.u_hat = np.zeros(grid_size, dtype=np.complex128)

    def __call__(self, problem, out) -> None:
        state = problem.origin
        alpha = problem.alpha
        rhs = problem.rhs
        delta = out[0]

        if rhs is None:
            delta.dv.fill(0.0)
        else:
            np.copyto(delta.dv, rhs[0].dv)

        if alpha == 0.0:
            if rhs is None:
                delta.du.fill(0.0)
            else:
                np.copyto(delta.du, rhs[0].du)
            return

        if rhs is None:
            np.copyto(self.u_hat, np.fft.fft(state.u))
        else:
            np.copyto(self.u_hat, np.fft.fft(state.u + rhs[0].du))

        self.u_hat /= 1.0 - alpha * self.operator_symbol
        delta.du[:] = np.fft.ifft(self.u_hat).real - state.u


def stark_layout(parameters: FitzHughNagumoParameters) -> Layout:
    shape = (parameters.grid_size,)
    return Layout(
        {
            "u": {"translation": "du", "shape": shape},
            "v": {"translation": "dv", "shape": shape},
        }
    )


def stark_configuration(
    scheme_type,
    parameters: FitzHughNagumoParameters,
) -> Configuration:
    exponent = 0.45
    if scheme_type is SchemeKennedyCarpenter43_7:
        exponent = 0.25
    elif scheme_type is SchemeKvaerno3:
        exponent = 1.0 / 3.0

    return Configuration(
        check_progress=False,
        scheme_tolerance=Tolerance(
            atol=parameters.tolerance_atol,
            rtol=parameters.tolerance_rtol,
        ),
        adaptive_scheme_safety=0.95,
        adaptive_scheme_error_exponent=exponent,
        resolvent_tolerance=Tolerance(
            atol=parameters.resolution_atol,
            rtol=parameters.resolution_rtol,
        ),
        resolvent_maximum_steps=parameters.resolution_max_iterations,
    )


def stark_ivp(
    parameters: FitzHughNagumoParameters,
    *,
    method: Method,
    initial_values: dict[str, np.ndarray],
    derivative: object | None = None,
):
    system = System(
        derivative=full_derivative(parameters) if derivative is None else derivative,
        layout=stark_layout(parameters),
    )
    return system.ivp(
        initial=initial_values,
        interval=Interval(parameters.t_start, parameters.initial_step, parameters.t_stop),
        method=method,
        engine=EngineNumpy,
        configuration=stark_configuration(method.scheme, parameters),
    )


def stark_solver(
    name: str,
    method: Method,
    parameters: FitzHughNagumoParameters,
    initial_conditions,
    reference,
    *,
    derivative: object | None = None,
):
    ivp = stark_ivp(
        parameters,
        method=method,
        derivative=derivative,
        initial_values=initial_conditions,
    )

    def solve_once() -> dict[str, Any]:
        result = ivp.final_result()
        return {
            "library": "STARK",
            "solver": name,
            "error": Comparison.fieldwise_rms_error(result.state, reference, ("u", "v")),
            "steps": result.steps,
        }

    return solve_once


def prepare_kvaerno3_anderson(
    problem_parameters,
    stark_parameters,
    initial_conditions,
    reference,
):
    parameters = FitzHughNagumoParameters(problem_parameters, stark_parameters)
    return stark_solver(
        "Kvaerno3 Anderson",
        Method(
            scheme=SchemeKvaerno3,
            resolvent=ResolventAnderson,
            resolvent_options={
                "depth": 4,
                "tableau": SchemeKvaerno3.tableau,
            },
        ),
        parameters,
        initial_conditions,
        reference,
    )


def prepare_kc43_imex_spectral(
    problem_parameters,
    stark_parameters,
    initial_conditions,
    reference,
):
    parameters = FitzHughNagumoParameters(problem_parameters, stark_parameters)
    return stark_solver(
        "KC43_7 IMEX Spectral",
        Method(
            scheme=SchemeKennedyCarpenter43_7,
            resolvent=FitzHughNagumoSpectralResolvent(
                parameters,
                tableau=SchemeKennedyCarpenter43_7.tableau,
            ),
        ),
        parameters,
        initial_conditions,
        reference,
        derivative=imex_derivative(parameters),
    )
