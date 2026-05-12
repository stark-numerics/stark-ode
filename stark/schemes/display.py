
"""Legacy scheme display helpers.

TODO: delete this module once resolvent-problem display helpers have moved to
`stark.schemes.support.display` or a more specific support module.
"""

from __future__ import annotations

from math import isclose

from stark.schemes.tableau import ButcherTableau, ImExButcherTableau


def _is_stiffly_accurate(tableau: ButcherTableau) -> bool:
    if not tableau.a:
        return False
    last_row = tableau.a[-1]
    if len(last_row) != len(tableau.b):
        return False
    return all(isclose(left, right, rel_tol=0.0, abs_tol=1.0e-14) for left, right in zip(last_row, tableau.b, strict=True))


def _is_lower_triangular(tableau: ButcherTableau) -> bool:
    for row_index, row in enumerate(tableau.a):
        for column_index in range(row_index + 1, len(row)):
            if not isclose(row[column_index], 0.0, rel_tol=0.0, abs_tol=1.0e-14):
                return False
    return True


def display_implicit_resolvent_problem(tableau: ButcherTableau, short_name: str, full_name: str) -> str:
    stage_count = len(tableau.c)
    lines = [
        f"{short_name} resolvent problem ({full_name})",
        "",
        "Unknown stage block:",
        f"    Delta = (Delta_1, ..., Delta_{stage_count})",
        "",
        "Solve for Delta so that, for i = 1, ..., s,",
        "",
        "    Delta_i - h * sum_{j=1}^s a_{ij} f(t_n + c_j h, x_n + Delta_j) = 0",
        "",
        "Then advance with",
        "",
        "    x_{n+1} = x_n + h * sum_{j=1}^s b_j f(t_n + c_j h, x_n + Delta_j)",
        "",
        "Here a_{ij}, b_j, and c_j are the coefficients of the method's Butcher tableau.",
    ]

    if _is_stiffly_accurate(tableau):
        lines.extend(
            [
                "",
                "This tableau is stiffly accurate, so the final stage coincides with the step update:",
                "",
                f"    x_{{n+1}} = x_n + Delta_{stage_count}",
            ]
        )

    lines.extend(
        [
            "",
            (
                "Because this tableau is lower triangular, STARK can choose to resolve the stages sequentially "
                "as shifted one-stage problems."
                if _is_lower_triangular(tableau)
                else "Because this tableau is not lower triangular, the natural resolvent problem is a coupled block system."
            ),
            "",
            "A custom resolvent for this method must accept arguments `(out, alpha, rhs=None)` and overwrite `out` with the solution block of the equation above, stored as `Block(Translation, Translation, ...)`.",
        ]
    )
    return "\n".join(lines)


def display_imex_resolvent_problem(tableau: ImExButcherTableau, short_name: str, full_name: str) -> str:
    stage_count = len(tableau.c)
    lines = [
        f"{short_name} IMEX resolvent problem ({full_name})",
        "",
        "Unknown stage block:",
        f"    Delta = (Delta_1, ..., Delta_{stage_count})",
        "",
        "Solve for Delta so that, for i = 1, ..., s,",
        "",
        "    Delta_i"
        " - h * sum_{j=1}^s a^im_{ij} f_im(t_n + c_j h, x_n + Delta_j)"
        " - h * sum_{j=1}^s a^ex_{ij} f_ex(t_n + c_j h, x_n + Delta_j)"
        " = 0",
        "",
        "Then advance with",
        "",
        "    x_{n+1}"
        " = x_n"
        " + h * sum_{j=1}^s b^im_j f_im(t_n + c_j h, x_n + Delta_j)"
        " + h * sum_{j=1}^s b^ex_j f_ex(t_n + c_j h, x_n + Delta_j)",
        "",
        "Here a^im_{ij}, a^ex_{ij}, b^im_j, b^ex_j, and c_j come from the IMEX Butcher tableau pair.",
        "",
        (
            "Because the implicit tableau is lower triangular, STARK can choose to resolve the implicit stage "
            "corrections sequentially."
            if _is_lower_triangular(tableau.implicit)
            else "Because the implicit tableau is not lower triangular, the natural IMEX resolvent problem is a coupled block system."
        ),
        "",
        "A custom resolvent for this method must accept arguments `(out, alpha, rhs=None)` and overwrite `out` with the solution block of the equation above, stored as `Block(Translation, Translation, ...)`.",
    ]
    return "\n".join(lines)


__all__ = ["display_implicit_resolvent_problem", "display_imex_resolvent_problem"]

