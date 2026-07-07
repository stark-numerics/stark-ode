"""Choose which fields contribute to adaptive-error norms."""

from __future__ import annotations

from typing import Any

import numpy as np

from stark import Frame, Field
from stark.engines import EngineNumpy
from stark.problem import NormExcluded, NormRMS


frame = Frame(
    (
        Field("u", translation="du", shape=(2,)),
        Field("diagnostic", translation="ddiagnostic", shape=(2,)),
    ),
    norms=(NormRMS(), NormExcluded()),
)


if __name__ == "__main__":
    engine = EngineNumpy(frame)
    delta: Any = engine.allocator.allocate_translation()
    delta.du[:] = np.array([3.0, 4.0])
    delta.ddiagnostic[:] = np.array([1000.0, 1000.0])

    print("Frame norm policy")
    print(f"physical field du:       {delta.du}")
    print(f"diagnostic field ignored: {delta.ddiagnostic}")
    print(f"translation norm:         {delta.norm():.6f}")
    print("Only du contributes; ddiagnostic is carried but excluded from norms.")
