# NumPy Engine Design Notes

`EngineNumpy` is the default backend for ordinary high-level STARK use.

It is intentionally boring: shaped frame fields become NumPy arrays, generated
Algebraist kernels operate on those arrays, and Numba is used by default when
it is installed.

## Role

The NumPy engine should be:

```text
predictable      easy to inspect and debug
fast enough      accelerated by generated kernels when possible
representative   the reference shape for other array backends
```

When a feature works on NumPy but not on another backend, treat NumPy as the
behavioural reference unless the feature is explicitly backend-specific.

## Accelerator Policy

`EngineNumpy` attempts to use `AcceleratorNumba` by default. If Numba is not
installed, it falls back to `AcceleratorNone` and the engine `repr` should make
that performance caveat visible.

This warning is an inspection point, not a runtime error. The package should
still work without Numba.

## Design Rule

Do not make NumPy special by hiding behaviour here that other backends cannot
reasonably mirror. NumPy is the default backend, but it is still one backend in
the engine family.
