from stark.integrate import Integrator
from stark.safety import Safety


class DummyInterval:
    def __init__(self, present: float, step: float, stop: float) -> None:
        self.present = present
        self.step = step
        self.stop = stop

    def copy(self) -> "DummyInterval":
        return DummyInterval(self.present, self.step, self.stop)

    def increment(self, dt: float) -> None:
        self.present += dt


class MarcherByStep:
    def __call__(self, interval: DummyInterval, state: object) -> None:
        if isinstance(state, dict):
            state["value"] += interval.step
        interval.increment(interval.step)

    def snapshot_state(self, state: object) -> object:
        if isinstance(state, dict):
            return dict(state)
        return state


class StalledMarcher:
    def __call__(self, interval: DummyInterval, state: object) -> None:
        del interval, state

    def snapshot_state(self, state: object) -> object:
        return state


class MarcherByClampedStep:
    def __call__(self, interval: DummyInterval, state: object) -> None:
        dt = min(interval.step, interval.stop - interval.present)
        if isinstance(state, dict):
            state["value"] += dt
        interval.increment(dt)

    def snapshot_state(self, state: object) -> object:
        if isinstance(state, dict):
            return dict(state)
        return state


class MarcherAndZeroNextStep(MarcherByClampedStep):
    def __call__(self, interval: DummyInterval, state: object) -> None:
        super().__call__(interval, state)
        if interval.present == interval.stop:
            interval.step = 0.0


def test_integrate_with_safety_rails_advances() -> None:
    integrate = Integrator()
    iterator = integrate(MarcherByStep(), DummyInterval(0.0, 0.25, 1.0), {"value": 0.0})
    history = list(iterator)

    assert len(history) == 4
    assert history[-1][0].present == 1.0
    assert history[0][0] is not history[-1][0]
    assert history[0][1] is not history[-1][1]
    assert history[0][1]["value"] == 0.25
    assert history[-1][1]["value"] == 1.0


def test_integrate_with_safety_rails_raises_on_no_progress() -> None:
    iterator = Integrator(safety=Safety())(StalledMarcher(), DummyInterval(0.0, 0.25, 1.0), object())

    try:
        next(iterator)
    except RuntimeError as exc:
        assert "no progress" in str(exc).lower()
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected safety rails to reject a stalled integration.")


def test_integrate_without_safety_rails_can_skip_progress_check() -> None:
    iterator = Integrator(safety=Safety.fast()).live(
        StalledMarcher(),
        DummyInterval(0.0, 0.25, 1.0),
        object(),
    )

    interval, _state = next(iterator)
    assert interval.present == 0.0


def test_integrate_live_yields_mutating_objects() -> None:
    integrate = Integrator()
    interval = DummyInterval(0.0, 0.25, 1.0)
    state = {"value": 0.0}
    iterator = integrate.live(MarcherByStep(), interval, state)

    first_interval, first_state = next(iterator)
    second_interval, second_state = next(iterator)

    assert first_interval is interval
    assert second_interval is interval
    assert first_state is state
    assert second_state is state
    assert first_state["value"] == 0.5
    assert second_state["value"] == 0.5


def test_integrate_snapshot_yields_integer_checkpoints() -> None:
    integrate = Integrator()
    interval = DummyInterval(0.0, 0.3, 1.0)
    state = {"value": 0.0}

    history = list(integrate(MarcherByClampedStep(), interval, state, checkpoints=4))

    assert [round(item[0].present, 12) for item in history] == [0.25, 0.5, 0.75, 1.0]
    assert [round(item[1]["value"], 12) for item in history] == [0.25, 0.5, 0.75, 1.0]
    assert history[0][0] is not interval
    assert history[0][1] is not state
    assert interval.stop == 1.0


def test_integrate_live_yields_only_explicit_checkpoints_and_final_stop() -> None:
    integrate = Integrator()
    interval = DummyInterval(0.0, 0.2, 1.0)
    state = {"value": 0.0}

    observed = []
    for checkpoint_interval, checkpoint_state in integrate.live(
        MarcherByClampedStep(),
        interval,
        state,
        checkpoints=[0.35, 0.8],
    ):
        observed.append((round(checkpoint_interval.present, 12), round(checkpoint_state["value"], 12)))
        assert checkpoint_interval is interval
        assert checkpoint_state is state

    assert observed == [(0.35, 0.35), (0.8, 0.8), (1.0, 1.0)]
    assert interval.stop == 1.0


def test_integrate_checkpoints_reuse_positive_step_after_exact_landing() -> None:
    integrate = Integrator()
    interval = DummyInterval(0.0, 0.3, 1.0)
    state = {"value": 0.0}

    observed = [
        round(checkpoint_interval.present, 12)
        for checkpoint_interval, _state in integrate.live(
            MarcherAndZeroNextStep(),
            interval,
            state,
            checkpoints=4,
        )
    ]

    assert observed == [0.25, 0.5, 0.75, 1.0]
    assert state["value"] == 1.0
    assert interval.step == 0.0


def test_integrate_rejects_unordered_checkpoints() -> None:
    integrate = Integrator()
    iterator = integrate.live(
        MarcherByClampedStep(),
        DummyInterval(0.0, 0.2, 1.0),
        {"value": 0.0},
        checkpoints=[0.5, 0.25],
    )

    try:
        next(iterator)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected unordered checkpoints to be rejected.")
