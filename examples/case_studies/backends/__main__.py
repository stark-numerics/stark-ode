"""Run the backend case-study lessons."""

from __future__ import annotations

from examples.case_studies.backends import lesson_01_numpy, lesson_02_jax, lesson_03_cupy, lesson_04_compare


def main() -> None:
    print("Backend case study")
    print("==================")
    print()
    lesson_01_numpy.main()
    print("\n" + "=" * 80 + "\n")
    lesson_02_jax.main()
    print("\n" + "=" * 80 + "\n")
    lesson_03_cupy.main()
    print("\n" + "=" * 80 + "\n")
    lesson_04_compare.main()


if __name__ == "__main__":
    main()
