"""Choose which fields contribute to adaptive-error norms."""

from __future__ import annotations

import numpy as np

from stark import Frame, FrameField
from stark.engines import EngineNumpy
from stark.problem import FrameNormExcluded, FrameNormRMS


frame = Frame(
    (
        FrameField("u", translation="du", shape=(2,), norm=FrameNormRMS()),
        FrameField("diagnostic", translation="ddiagnostic", shape=(2,), norm=FrameNormExcluded()),
    )
)


def main() -> None:
    engine = EngineNumpy(frame)
    delta = engine.allocator.allocate_translation()
    delta.du[:] = np.array([3.0, 4.0])
    delta.ddiagnostic[:] = np.array([1000.0, 1000.0])

    print("Frame norm policy")
    print(f"physical field du:       {delta.du}")
    print(f"diagnostic field ignored: {delta.ddiagnostic}")
    print(f"translation norm:         {delta.norm():.6f}")
    print("Only du contributes; ddiagnostic is carried but excluded from norms.")


if __name__ == "__main__":
    main()
