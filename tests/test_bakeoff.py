from stark.comparison import Comparator, ComparatorEntry, ComparatorProblem


class DummyInterval:
    def __init__(self, present: float, step: float, stop: float) -> None:
        self.present = present
        self.step = step
        self.stop = stop

    def copy(self) -> "DummyInterval":
        return DummyInterval(self.present, self.step, self.stop)

    def increment(self, dt: float) -> None:
        self.present += dt


class WeightedMarcher:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, interval: DummyInterval, state: dict[str, float]) -> None:
        state["value"] += self.weight * interval.step
        interval.increment(interval.step)

    def snapshot_state(self, state: dict[str, float]) -> dict[str, float]:
        return dict(state)


def test_bakeoff_reports_pairwise_differences() -> None:
    problem = ComparatorProblem(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
        diagnostics=lambda state: {"value": state["value"]},
    )
    entries = [
        ComparatorEntry("baseline", lambda: WeightedMarcher(1.0), metadata={"variant": "baseline", "accelerator": "none"}),
        ComparatorEntry("slow", lambda: WeightedMarcher(0.5), metadata={"variant": "slow", "accelerator": "none"}),
    ]

    report = Comparator(problem, entries, repeats=2)()

    assert len(report.results) == 2
    assert report.results[0].steps == 4
    assert report.results_by_name()["baseline"].timing.median >= 0.0
    assert report.diagnostics_by_name()["baseline"]["value"] == 1.0
    assert report.metadata_by_name()["baseline"]["variant"] == "baseline"
    assert report.final_difference_map()["baseline"]["baseline"] == 0.0
    assert report.final_difference_map()["baseline"]["slow"] > 0.0
    assert report.final_difference_map()["slow"]["baseline"] > 0.0
    assert "Pairwise final-state differences" in report.render()
    assert "Configuration Table" in report.render()
    assert "Diagnostics Table" in report.render()
    assert "Custom entry: the profile buckets are approximate" in report.render()
    assert "Custom entry hotspots" in report.render()
    assert "test_bakeoff.py" in report.render()
    assert "baseline" in report.render()
    assert "slow" in report.render()


def test_bakeoff_accepts_direct_marchers() -> None:
    problem = ComparatorProblem(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )
    entries = [
        ComparatorEntry("baseline", WeightedMarcher(1.0)),
        ComparatorEntry("slow", lambda: WeightedMarcher(0.5)),
    ]

    report = Comparator(problem, entries, repeats=1)()

    assert report.final_difference_map()["baseline"]["baseline"] == 0.0
    assert report.final_difference_map()["baseline"]["slow"] > 0.0


def test_bakeoff_reports_pairwise_trajectory_differences() -> None:
    problem = ComparatorProblem(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
        checkpoints=2,
    )
    entries = [
        ComparatorEntry("baseline", lambda: WeightedMarcher(1.0), metadata={"variant": "baseline"}),
        ComparatorEntry("slow", lambda: WeightedMarcher(0.5), metadata={"variant": "slow"}),
    ]

    report = Comparator(problem, entries, repeats=1)()

    assert report.trajectory_differences is not None
    assert report.trajectory_difference_map() is not None
    assert report.trajectory_difference_map()["baseline"]["baseline"] == 0.0
    assert report.trajectory_difference_map()["baseline"]["slow"] > 0.0
    assert "Pairwise trajectory differences" in report.render()
    assert "default RMS of checkpoint-wise state differences" in report.render()


def test_bakeoff_skips_diagnostics_table_when_problem_has_none() -> None:
    problem = ComparatorProblem(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )
    entries = [
        ComparatorEntry("baseline", lambda: WeightedMarcher(1.0)),
        ComparatorEntry("slow", lambda: WeightedMarcher(0.5)),
    ]

    report = Comparator(problem, entries, repeats=1)()

    assert "Diagnostics Table" not in report.render()


def test_bakeoff_requires_two_entries() -> None:
    problem = ComparatorProblem(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )

    try:
        Comparator(problem, [ComparatorEntry("only", lambda: WeightedMarcher(1.0))])
    except ValueError as exc:
        assert "at least two entries" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected bakeoff to require at least two entries.")











