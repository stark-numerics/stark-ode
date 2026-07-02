from __future__ import annotations

from stark import Method, MethodError


class ExplicitScheme:
    def __init__(self, derivative, allocator) -> None:
        del derivative, allocator


class ImplicitScheme:
    def __init__(self, derivative, allocator, resolvent) -> None:
        del derivative, allocator, resolvent


class PicardResolvent:
    def __init__(self, allocator) -> None:
        del allocator


class NewtonResolvent:
    def __init__(self, allocator, linearizer, inverter) -> None:
        del allocator, linearizer, inverter


class Inverter:
    pass


def test_stark_method_accepts_explicit_scheme_recipe() -> None:
    method = Method(scheme=ExplicitScheme)

    assert method.scheme is ExplicitScheme
    assert method.resolvent is None
    assert method.inverter is None


def test_stark_method_requires_resolvent_for_implicit_scheme() -> None:
    try:
        Method(scheme=ImplicitScheme)
    except MethodError as exc:
        assert "requires a resolvent" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected implicit method without resolvent to fail.")


def test_stark_method_accepts_implicit_scheme_recipe() -> None:
    method = Method(scheme=ImplicitScheme, resolvent=PicardResolvent)

    assert method.scheme is ImplicitScheme
    assert method.resolvent is PicardResolvent


def test_stark_method_accepts_ready_resolvent_instance() -> None:
    resolvent = PicardResolvent(allocator=object())
    method = Method(scheme=ImplicitScheme, resolvent=resolvent)

    assert method.resolvent is resolvent


def test_stark_method_rejects_options_for_ready_resolvent_instance() -> None:
    resolvent = PicardResolvent(allocator=object())
    try:
        Method(
            scheme=ImplicitScheme,
            resolvent=resolvent,
            resolvent_options={"depth": 4},
        )
    except MethodError as exc:
        assert "resolvent_options require a resolvent class" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected ready resolvent options to fail.")


def test_stark_method_requires_inverter_for_linearized_resolvent() -> None:
    try:
        Method(scheme=ImplicitScheme, resolvent=NewtonResolvent)
    except MethodError as exc:
        assert "requires an inverter" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected linearized resolvent without inverter to fail.")


def test_stark_method_accepts_linearized_method_recipe() -> None:
    method = Method(
        scheme=ImplicitScheme,
        resolvent=NewtonResolvent,
        inverter=Inverter,
    )

    assert method.inverter is Inverter
