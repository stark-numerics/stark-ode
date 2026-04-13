from __future__ import annotations

from stark.contracts import Block, Derivative, Linearizer, State
from stark.scheme_support.workspace import SchemeWorkspace


class ShiftedJacobianOperator:
    """Mutable operator configured by a linearizer for one implicit solve."""

    __slots__ = ("apply", "method_name")

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name
        self.apply = self._unconfigured

    def __call__(self, out, translation) -> None:
        self.apply(out, translation)

    def _unconfigured(self, out, translation) -> None:
        del out, translation
        raise RuntimeError(f"{self.method_name} Jacobian operator was used before the linearizer configured it.")


class ShiftedResidualOperator:
    """Linearized operator `I - alpha J` for one shifted implicit residual."""

    __slots__ = ("combine2", "jacobian_buffer", "jacobian", "alpha")

    def __init__(self, workspace: SchemeWorkspace, jacobian: ShiftedJacobianOperator) -> None:
        self.combine2 = workspace.combine2
        self.jacobian_buffer = workspace.allocate_translation()
        self.jacobian = jacobian
        self.alpha = 0.0

    def __call__(self, out, translation) -> None:
        self.jacobian(self.jacobian_buffer, translation)
        self.combine2(out, 1.0, translation, -self.alpha, self.jacobian_buffer)


class ShiftedImplicitResidual:
    """
    Residual worker for equations of the form

        delta - alpha f(state + known_shift + delta) = 0.

    This covers backward Euler, singly diagonally implicit Runge-Kutta stages,
    and variable-step BDF2 corrections. When a linearizer is supplied the
    associated Newton operator is `I - alpha J`.
    """

    __slots__ = (
        "method_name",
        "scale",
        "combine2",
        "copy_state",
        "base_state",
        "known_state",
        "trial_state",
        "known_shift",
        "derivative",
        "derivative_buffer",
        "alpha",
        "linearizer",
        "jacobian_operator",
        "residual_operator",
        "_linearize",
    )

    def __init__(
        self,
        method_name: str,
        derivative: Derivative,
        workspace: SchemeWorkspace,
        linearizer: Linearizer | None = None,
    ) -> None:
        self.method_name = method_name
        self.scale = workspace.scale
        self.combine2 = workspace.combine2
        self.copy_state = workspace.copy_state
        self.base_state = workspace.allocate_state_buffer()
        self.known_state = workspace.allocate_state_buffer()
        self.trial_state = workspace.allocate_state_buffer()
        self.known_shift = workspace.allocate_translation()
        self.derivative = derivative
        self.derivative_buffer = workspace.allocate_translation()
        self.alpha = 0.0
        self.linearizer = linearizer
        self.jacobian_operator = ShiftedJacobianOperator(method_name)
        self.residual_operator = ShiftedResidualOperator(workspace, self.jacobian_operator)
        self._linearize = self._linearize_configured if linearizer is not None else self._linearize_missing

    def configure(self, state: State, alpha: float, known_shift=None) -> None:
        self.copy_state(self.base_state, state)
        self.alpha = alpha
        if known_shift is None:
            self.scale(self.known_shift, 0.0, self.known_shift)
            return
        self.combine2(self.known_shift, 0.0, self.known_shift, 1.0, known_shift)

    def __call__(self, out: Block, block: Block) -> None:
        delta = block[0]
        self.known_shift(self.base_state, self.known_state)
        delta(self.known_state, self.trial_state)
        self.derivative(self.trial_state, self.derivative_buffer)
        self.combine2(out[0], 1.0, delta, -self.alpha, self.derivative_buffer)

    def linearize(self, out, block: Block) -> None:
        self._linearize(out, block)

    def _linearize_missing(self, out, block: Block) -> None:
        del out, block
        raise RuntimeError(f"{self.method_name} Newton resolution requires a linearizer.")

    def _linearize_configured(self, out, block: Block) -> None:
        linearizer = self.linearizer
        assert linearizer is not None
        self.known_shift(self.base_state, self.known_state)
        block[0](self.known_state, self.trial_state)
        linearizer(self.jacobian_operator, self.trial_state)
        self.residual_operator.alpha = self.alpha
        out.operators[0] = self.residual_operator
__all__ = [
    "ShiftedImplicitResidual",
    "ShiftedJacobianOperator",
    "ShiftedResidualOperator",
]
