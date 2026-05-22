from stark.algebraist.classic import Algebraist, AlgebraistField


def test_algebraist_scope_stops_before_resolvent_and_inverter_generation() -> None:
    algebraist = Algebraist(fields=(AlgebraistField("value", "value"),))

    assert not hasattr(algebraist, "bind_resolvent")
    assert not hasattr(algebraist, "bind_inverter")
    assert not hasattr(algebraist, "bind_preconditioner")

