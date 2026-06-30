# CuPy Engine Design Notes

`EngineCupy` stores shaped frame fields in CuPy arrays and emits CuPy-native
generated algebra.

The key performance rule is simple: do not drive GPU arrays through Python
scalar loops. CuPy needs whole-array or fused kernels.

## Target Policy

CuPy uses `AlgebraistGeneratorTargetCupy`, which emits CuPy expressions and
`cupy.ElementwiseKernel` wrappers for repeated algebra. This target exists
because the generic generated Python-loop target is the wrong shape for GPU
arrays.

Norms and inner products should be expressed as CuPy reductions. Timing code
must account for GPU synchronization if it wants honest elapsed times.

## Accelerator Policy

`EngineCupy` currently defaults to `AcceleratorNone`. That does not mean the
engine is unaccelerated in the same sense as unaccelerated CPU NumPy: the CuPy
target itself emits GPU work. A separate accelerator would only be useful if it
fuses or compiles work beyond what the CuPy target already does.

## Hint Types

`HintCupyArray` should describe the exact CuPy array surface used by CuPy
carriers, such as indexing, reshape, scalar extraction, and elementwise
arithmetic. Expand that hint when carrier code legitimately uses more of CuPy;
do not replace it with broad `Any` or rename it into a public STARK object.

## Design Rule

Keep CuPy-specific expression choices local to this backend. If an optimisation
only makes sense because data lives on a GPU, it probably belongs here rather
than in shared Algebraist code.
