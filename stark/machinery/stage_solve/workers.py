from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
from stark.auditor import Auditor
from stark.contracts import Block, Derivative, ImExDerivative, IntervalLike, Resolvent, State, Translation, Workbench
from stark.resolvents.support.guard import ResolventTableauGuard
from stark.machinery.stage_solve.workspace import SchemeWorkspace


def _require_resolvent(method_name: str, resolvent: Resolvent | None) -> Resolvent:
    if resolvent is None:
        raise TypeError(f"{method_name} requires an explicit resolvent.")
    return resolvent


class _ImExStageSolver:
    """Bind a shifted stage state and resolve the diagonal implicit correction."""

    __slots__ = ("copy_state", "apply_delta", "scale", "resolvent", "stage_state")

    def __init__(self, workspace: SchemeWorkspace, resolvent: Resolvent) -> None:
        self.copy_state = workspace.copy_state
        self.apply_delta = workspace.apply_delta
        self.scale = workspace.scale
        self.resolvent = resolvent
        self.stage_state = workspace.allocate_state_buffer()

    def __call__(
        self,
        interval: IntervalLike,
        base_state: State,
        shift: Translation | None,
        alpha: float,
        block: Block,
    ) -> State:
        if shift is None:
            self.copy_state(self.stage_state, base_state)
        else:
            shift(base_state, self.stage_state)

        block.items[0] = self.scale(block[0], 0.0, block[0])
        if alpha != 0.0:
            self.resolvent.bind(interval, self.stage_state)
            self.resolvent(block, alpha)
            self.apply_delta(block[0], self.stage_state)
        return self.stage_state


class ImExStepper:
    """Sequential IMEX Runge-Kutta worker for one additive tableau pair."""

    __slots__ = (
        "tableau",
        "explicit_derivative",
        "implicit_derivative",
        "workspace",
        "stage_solver",
        "explicit_rates",
        "implicit_rates",
        "stage_blocks",
        "shift",
        "trial",
        "error",
        "shift_coefficients",
        "shift_translations",
        "weight_coefficients",
        "weight_translations",
    )

    def __init__(self, derivative: ImExDerivative, workspace: SchemeWorkspace, resolvent: Resolvent, tableau) -> None:
        self.tableau = tableau
        self.explicit_derivative = BoundDerivative(derivative.explicit)
        self.implicit_derivative = BoundDerivative(derivative.implicit)
        self.workspace = workspace
        self.stage_solver = _ImExStageSolver(workspace, resolvent)
        stage_count = len(tableau.c)
        self.explicit_rates = list(workspace.allocate_translation_buffers(stage_count))
        self.implicit_rates = list(workspace.allocate_translation_buffers(stage_count))
        stage_deltas = workspace.allocate_translation_buffers(stage_count)
        self.stage_blocks = [Block([delta]) for delta in stage_deltas]
        self.shift, self.trial, self.error = workspace.allocate_translation_buffers(3)
        max_term_count = 2 * stage_count
        self.shift_coefficients = [0.0] * max_term_count
        self.shift_translations = [self.shift] * max_term_count
        self.weight_coefficients = [0.0] * max_term_count
        self.weight_translations = [self.shift] * max_term_count

    def __repr__(self) -> str:
        return f"ImExStepper(stages={len(self.tableau.c)!r})"

    __str__ = __repr__

    def step(self, interval: IntervalLike, state: State, dt: float, *, include_norms: bool = False):
        stage_interval = self.workspace.stage_interval
        explicit = self.explicit_derivative
        implicit = self.implicit_derivative
        explicit_a = self.tableau.explicit.a
        implicit_a = self.tableau.implicit.a
        shift_coefficients = self.shift_coefficients
        shift_translations = self.shift_translations

        for stage_index, c_value in enumerate(self.tableau.c):
            shift_count = 0
            explicit_row = explicit_a[stage_index]
            implicit_row = implicit_a[stage_index]

            for source_index in range(stage_index):
                if source_index < len(explicit_row):
                    coefficient = explicit_row[source_index]
                    if coefficient != 0.0:
                        shift_coefficients[shift_count] = dt * coefficient
                        shift_translations[shift_count] = self.explicit_rates[source_index]
                        shift_count += 1
                if source_index < len(implicit_row):
                    coefficient = implicit_row[source_index]
                    if coefficient != 0.0:
                        shift_coefficients[shift_count] = dt * coefficient
                        shift_translations[shift_count] = self.implicit_rates[source_index]
                        shift_count += 1

            shift = self._accumulate_terms(self.shift, shift_count, shift_coefficients, shift_translations) if shift_count else None
            alpha = dt * implicit_row[stage_index] if stage_index < len(implicit_row) else 0.0
            current_interval = stage_interval(interval, dt, c_value * dt)
            stage_state = self.stage_solver(current_interval, state, shift, alpha, self.stage_blocks[stage_index])
            implicit(current_interval, stage_state, self.implicit_rates[stage_index])
            explicit(current_interval, stage_state, self.explicit_rates[stage_index])

        high = self._combine_weights(self.trial, dt, self.tableau.explicit.b, self.tableau.implicit.b)
        if self.tableau.embedded_order is None:
            if include_norms:
                return high, None, high.norm(), None
            return high, None, None, None
        explicit_low = self.tableau.explicit.b_embedded
        implicit_low = self.tableau.implicit.b_embedded
        assert explicit_low is not None
        assert implicit_low is not None
        low = self._combine_weights(self.error, dt, explicit_low, implicit_low)
        error = self.workspace.combine2(self.error, 1.0, high, -1.0, low)
        if include_norms:
            return high, error, high.norm(), error.norm()
        return high, error, None, None

    def _combine_weights(self, out: Translation, dt: float, explicit_weights, implicit_weights) -> Translation:
        term_count = 0
        weight_coefficients = self.weight_coefficients
        weight_translations = self.weight_translations
        for coefficient, rate in zip(explicit_weights, self.explicit_rates, strict=True):
            if coefficient != 0.0:
                weight_coefficients[term_count] = dt * coefficient
                weight_translations[term_count] = rate
                term_count += 1
        for coefficient, rate in zip(implicit_weights, self.implicit_rates, strict=True):
            if coefficient != 0.0:
                weight_coefficients[term_count] = dt * coefficient
                weight_translations[term_count] = rate
                term_count += 1
        return self._accumulate_terms(out, term_count, weight_coefficients, weight_translations)

    def _accumulate_terms(
        self,
        out: Translation,
        term_count: int,
        coefficients: list[float],
        translations: list[Translation],
    ) -> Translation:
        workspace = self.workspace
        if term_count == 0:
            return workspace.scale(out, 0.0, out)

        if term_count == 1:
            return workspace.scale(out, coefficients[0], translations[0])

        if term_count <= 12:
            terms = []
            for index in range(term_count):
                terms.append(coefficients[index])
                terms.append(translations[index])
            return getattr(workspace, f"combine{term_count}")(out, *terms)

        total = workspace.scale(out, coefficients[0], translations[0])
        combine2 = workspace.combine2
        for index in range(1, term_count):
            total = combine2(out, 1.0, total, coefficients[index], translations[index])
        return total


class ShiftedOneStageResolventStep:
    """Reusable bind/zero/solve worker for one-stage shifted implicit updates."""

    __slots__ = ("workspace", "tableau_guard", "resolvent", "trial_block")

    def __init__(self, method_name: str, tableau, derivative: Derivative, workbench: Workbench, resolvent: Resolvent) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.tableau_guard = ResolventTableauGuard(method_name, tableau)
        self.resolvent = _require_resolvent(method_name, resolvent)
        self.tableau_guard(self.resolvent)
        self.trial_block = Block([translation_probe])

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    def solve(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        *,
        alpha: float,
        stage_shift: float,
        rhs: Block | None = None,
    ) -> Translation:
        workspace = self.workspace
        self.trial_block.items[0] = workspace.scale(self.trial_block[0], 0.0, self.trial_block[0])
        self.resolvent.bind(workspace.stage_at(interval, dt, stage_shift), state)
        self.resolvent(self.trial_block, alpha, rhs=rhs)
        return self.trial_block[0]


class CoupledCollocationResolventStep:
    """Reusable bind/zero/solve worker for fully coupled collocation stages."""

    __slots__ = ("workspace", "tableau_guard", "resolvent", "stage_block")

    def __init__(
        self,
        method_name: str,
        tableau,
        derivative: Derivative,
        workbench: Workbench,
        stage_count: int,
        resolvent: Resolvent,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.tableau_guard = ResolventTableauGuard(method_name, tableau)
        self.resolvent = _require_resolvent(method_name, resolvent)
        self.tableau_guard(self.resolvent)
        self.stage_block = Block([self.workspace.allocate_translation() for _ in range(stage_count)])

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    def solve(self, interval: IntervalLike, state: State, dt: float) -> Block:
        workspace = self.workspace
        self.resolvent.bind(interval, state)
        for index, item in enumerate(self.stage_block):
            self.stage_block.items[index] = workspace.scale(item, 0.0, item)
        self.resolvent(self.stage_block, dt)
        return self.stage_block


class SequentialDIRKResolventStep:
    """Reusable stage worker for sequential DIRK and ESDIRK families."""

    __slots__ = ("workspace", "tableau_guard", "resolvent", "stage_state", "stage_blocks")

    def __init__(
        self,
        method_name: str,
        tableau,
        derivative: Derivative,
        workbench: Workbench,
        implicit_stage_count: int,
        resolvent: Resolvent,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.tableau_guard = ResolventTableauGuard(method_name, tableau)
        self.resolvent = _require_resolvent(method_name, resolvent)
        self.tableau_guard(self.resolvent)
        self.stage_state = self.workspace.allocate_state_buffer()
        self.stage_blocks = tuple(Block([self.workspace.allocate_translation()]) for _ in range(implicit_stage_count))

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    def solve(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        *,
        block_index: int,
        stage_shift: float,
        alpha: float,
        known_shift: Translation | None = None,
        out: Translation | None = None,
    ) -> Translation:
        workspace = self.workspace
        stage_state = state
        if known_shift is not None:
            known_shift(state, self.stage_state)
            stage_state = self.stage_state
        block = self.stage_blocks[block_index]
        target = block[0] if out is None else out
        block.items[0] = workspace.scale(target, 0.0, target)
        self.resolvent.bind(workspace.stage_at(interval, dt, stage_shift), stage_state)
        self.resolvent(block, alpha)
        return block[0]


__all__ = [
    "CoupledCollocationResolventStep",
    "ImExStepper",
    "SequentialDIRKResolventStep",
    "ShiftedOneStageResolventStep",
]


