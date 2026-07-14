from __future__ import annotations

from stark.engines.accelerators import AcceleratorNone


def test_none_accelerator_compile_accepts_decorator_form() -> None:
    accelerator = AcceleratorNone()

    @accelerator.compile
    def worker(value: float) -> float:
        return 2.0 * value

    assert worker(3.0) == 6.0


def test_none_accelerator_compile_accepts_configured_decorator_form() -> None:
    accelerator = AcceleratorNone()

    @accelerator.compile(label="audit")
    def worker(value: float) -> float:
        return 3.0 * value

    assert worker(4.0) == 12.0


def test_none_accelerator_compile_examples_leaves_callable_unchanged() -> None:
    accelerator = AcceleratorNone()

    def worker(value: float) -> float:
        return 4.0 * value

    compiled = accelerator.compile_examples(worker, 2.0)

    assert compiled is worker
    assert compiled(5.0) == 20.0
