from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def _format_entry(value: float) -> str:
    if value == 0.0:
        return "0"

    fraction = Fraction(value).limit_denominator(256)
    if abs(float(fraction) - value) < 1.0e-12:
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"

    return f"{value:.6g}"


@dataclass(frozen=True, slots=True)
class ButcherTableau:
    c: tuple[float, ...]
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    order: int
    b_embedded: tuple[float, ...] | None = None
    embedded_order: int | None = None
    short_name: str | None = None
    full_name: str | None = None

    def __repr__(self) -> str:
        name = " ".join(part for part in (self.short_name, self.full_name) if part).strip()
        name_text = f", name={name!r}" if name else ""
        return (
            "ButcherTableau("
            f"stages={len(self.c)}, "
            f"order={self.order!r}, "
            f"embedded_order={self.embedded_order!r}"
            f"{name_text})"
        )

    @property
    def b_high(self) -> tuple[float, ...]:
        return self.b

    @property
    def b_low(self) -> tuple[float, ...]:
        if self.b_embedded is None:
            raise AttributeError("This tableau has no embedded weights.")
        return self.b_embedded

    @property
    def high_order(self) -> int:
        return self.order

    @property
    def low_order(self) -> int:
        if self.embedded_order is None:
            raise AttributeError("This tableau has no embedded order.")
        return self.embedded_order

    def display(self, short_name: str | None = None, full_name: str | None = None) -> str:
        stage_count = len(self.c)
        rows: list[list[str]] = []

        for c_value, row in zip(self.c, self.a, strict=True):
            coefficients = [_format_entry(value) for value in row]
            coefficients.extend("" for _ in range(stage_count - len(coefficients)))
            rows.append([_format_entry(c_value), *coefficients])

        widths = [
            max(len(row[column]) for row in rows)
            for column in range(stage_count + 1)
        ]

        rendered_rows = []
        for row in rows:
            c_value = row[0].rjust(widths[0])
            coefficients = " ".join(
                value.rjust(width)
                for value, width in zip(row[1:], widths[1:], strict=True)
            )
            rendered_rows.append(f"{c_value} | {coefficients}".rstrip())

        resolved_short_name = short_name if short_name is not None else self.short_name
        resolved_full_name = full_name if full_name is not None else self.full_name
        name_prefix = " ".join(
            part for part in (resolved_short_name, resolved_full_name) if part
        ).strip()

        if self.b_embedded is None:
            weights = ["", *[_format_entry(value) for value in self.b]]
            widths = [
                max(len(row[column]) for row in [*rows, weights])
                for column in range(stage_count + 1)
            ]
            rendered_weights = " ".join(
                value.rjust(width)
                for value, width in zip(weights[1:], widths[1:], strict=True)
            )
            rendered_weights = f"{' ' * widths[0]} | {rendered_weights}".rstrip()
            divider_width = max(
                *(len(row) for row in rendered_rows),
                len(rendered_weights),
            )
            order_text = f"order {self.order}"
            heading = f"{name_prefix} Butcher tableau ({order_text})" if name_prefix else f"Butcher tableau ({order_text})"
            return "\n".join(
                [
                    heading,
                    *rendered_rows,
                    "-" * divider_width,
                    rendered_weights,
                ]
            )

        weight_width = max(len("b"), len("b_embedded"))
        rendered_high = " ".join(
            _format_entry(value).rjust(width)
            for value, width in zip(self.b, widths[1:], strict=True)
        )
        rendered_low = " ".join(
            _format_entry(value).rjust(width)
            for value, width in zip(self.b_embedded, widths[1:], strict=True)
        )
        rendered_high = f"{' ' * widths[0]} | {'b'.rjust(weight_width)} {rendered_high}".rstrip()
        rendered_low = f"{' ' * widths[0]} | {'b_embedded'.rjust(weight_width)} {rendered_low}".rstrip()
        divider_width = max(
            *(len(row) for row in rendered_rows),
            len(rendered_high),
            len(rendered_low),
        )
        order_text = f"orders {self.order}/{self.embedded_order}"
        heading = f"{name_prefix} Butcher tableau ({order_text})" if name_prefix else f"Butcher tableau ({order_text})"
        return "\n".join(
            [
                heading,
                *rendered_rows,
                "-" * divider_width,
                rendered_high,
                rendered_low,
            ]
        )

    def __str__(self) -> str:
        return self.display()


class EmbeddedButcherTableau(ButcherTableau):
    def __init__(
        self,
        *,
        c: tuple[float, ...],
        a: tuple[tuple[float, ...], ...],
        b_high: tuple[float, ...],
        b_low: tuple[float, ...],
        high_order: int,
        low_order: int,
        short_name: str | None = None,
        full_name: str | None = None,
    ) -> None:
        super().__init__(
            c=c,
            a=a,
            b=b_high,
            order=high_order,
            b_embedded=b_low,
            embedded_order=low_order,
            short_name=short_name,
            full_name=full_name,
        )


@dataclass(frozen=True, slots=True)
class ImExButcherTableau:
    explicit: ButcherTableau
    implicit: ButcherTableau
    short_name: str | None = None
    full_name: str | None = None

    def __post_init__(self) -> None:
        if len(self.explicit.c) != len(self.implicit.c):
            raise ValueError("IMEX explicit and implicit tableaus must have the same stage count.")
        if self.explicit.c != self.implicit.c:
            raise ValueError("IMEX explicit and implicit tableaus must share the same stage abscissae.")
        if self.explicit.order != self.implicit.order:
            raise ValueError("IMEX explicit and implicit tableaus must agree on method order.")
        if self.explicit.embedded_order != self.implicit.embedded_order:
            raise ValueError("IMEX explicit and implicit tableaus must agree on embedded order.")

    def __repr__(self) -> str:
        name = " ".join(part for part in (self.short_name, self.full_name) if part).strip()
        name_text = f", name={name!r}" if name else ""
        return (
            "ImExButcherTableau("
            f"stages={len(self.explicit.c)!r}, "
            f"order={self.order!r}, "
            f"embedded_order={self.embedded_order!r}"
            f"{name_text})"
        )

    @property
    def c(self) -> tuple[float, ...]:
        return self.explicit.c

    @property
    def order(self) -> int:
        return self.explicit.order

    @property
    def embedded_order(self) -> int | None:
        return self.explicit.embedded_order

    def display(self, short_name: str | None = None, full_name: str | None = None) -> str:
        resolved_short_name = short_name if short_name is not None else self.short_name
        resolved_full_name = full_name if full_name is not None else self.full_name
        name_prefix = " ".join(part for part in (resolved_short_name, resolved_full_name) if part).strip()
        order_text = (
            f"orders {self.order}/{self.embedded_order}"
            if self.embedded_order is not None
            else f"order {self.order}"
        )
        heading = f"{name_prefix} IMEX Butcher tableau ({order_text})" if name_prefix else f"IMEX Butcher tableau ({order_text})"
        return "\n".join(
            [
                heading,
                "[explicit]",
                self.explicit.display(),
                "[implicit]",
                self.implicit.display(),
            ]
        )

    def __str__(self) -> str:
        return self.display()


__all__ = ["ButcherTableau", "EmbeddedButcherTableau", "ImExButcherTableau"]









