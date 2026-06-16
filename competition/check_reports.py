from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


REPORT_MODULES = (
    "competition.allen_cahn_1d.report",
    "competition.brusselator_2d.report",
    "competition.fitzhugh_nagumo_1d.report",
    "competition.fput.report",
    "competition.hires.report",
    "competition.robertson.report",
)

COMPLETED_ROW = re.compile(
    r"^Completed (?P<library>.+?) (?P<solver>.*?): "
    r"steps=(?P<steps>[^,]+), "
    r"error=(?P<error>[0-9.eE+-]+), "
    r"setup=(?P<setup>[0-9.eE+-]+)s, "
    r"warmup=(?P<warmup>[0-9.eE+-]+)s, "
    r"median=(?P<median>[0-9.eE+-]+)s, "
    r"min=(?P<minimum>[0-9.eE+-]+)s$"
)


@dataclass(frozen=True, slots=True)
class Row:
    library: str
    solver: str
    steps: int | None
    error: float
    setup: float
    warmup: float
    median: float
    minimum: float

    @property
    def key(self) -> str:
        return f"{self.library} {self.solver}"

    @classmethod
    def from_match(cls, match: re.Match[str]) -> Row:
        steps_text = match.group("steps")
        return cls(
            library=match.group("library"),
            solver=match.group("solver"),
            steps=None if steps_text == "-" else int(steps_text),
            error=float(match.group("error")),
            setup=float(match.group("setup")),
            warmup=float(match.group("warmup")),
            median=float(match.group("median")),
            minimum=float(match.group("minimum")),
        )

    def as_json(self) -> dict[str, int | float | str | None]:
        return {
            "library": self.library,
            "solver": self.solver,
            "steps": self.steps,
            "error": self.error,
            "setup": self.setup,
            "warmup": self.warmup,
            "median": self.median,
            "minimum": self.minimum,
        }


@dataclass(frozen=True, slots=True)
class Report:
    module: str
    elapsed: float
    rows: tuple[Row, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "elapsed": self.elapsed,
            "rows": {row.key: row.as_json() for row in self.rows},
        }


def parse_rows(stdout: str) -> tuple[Row, ...]:
    rows = []
    for line in stdout.splitlines():
        match = COMPLETED_ROW.match(line)
        if match is not None:
            rows.append(Row.from_match(match))
    if rows:
        return tuple(rows)

    return parse_table_rows(stdout)


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.split("|")]


def parse_report_table(stdout: str, title: str) -> list[dict[str, str]]:
    lines = stdout.splitlines()
    rows: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        if line.strip() != title:
            continue

        header_index = index + 1
        separator_index = index + 2
        if separator_index >= len(lines):
            return rows

        headers = table_cells(lines[header_index])
        for row_line in lines[separator_index + 1 :]:
            if not row_line.strip():
                break
            if "-+-" in row_line:
                continue
            cells = table_cells(row_line)
            if len(cells) < len(headers):
                continue
            rows.append(dict(zip(headers, cells, strict=False)))
        return rows

    return rows


def seconds(text: str) -> float:
    return float(text.removesuffix("s"))


def parse_table_rows(stdout: str) -> tuple[Row, ...]:
    error_rows = parse_report_table(stdout, "Error Table")
    preparation_rows = parse_report_table(stdout, "Preparation Timing Table")
    timing_rows = parse_report_table(stdout, "Warm Run Timing Table")
    if not timing_rows:
        timing_rows = parse_report_table(stdout, "Run Timing Table")

    preparation_by_key = {
        (row["library"], row["solver"]): row
        for row in preparation_rows
    }
    timing_by_key = {
        (row["library"], row["solver"]): row
        for row in timing_rows
    }

    rows: list[Row] = []
    for error_row in error_rows:
        key = (error_row["library"], error_row["solver"])
        timing_row = timing_by_key.get(key)
        if timing_row is None:
            continue
        preparation_row = preparation_by_key.get(key, {})
        steps_text = error_row["steps"]
        rows.append(
            Row(
                library=error_row["library"],
                solver=error_row["solver"],
                steps=None if steps_text == "-" else int(steps_text),
                error=float(error_row["error"]),
                setup=seconds(preparation_row.get("setup", "0s")),
                warmup=seconds(preparation_row.get("warmup", "0s")),
                median=seconds(timing_row["median"]),
                minimum=seconds(timing_row["min"]),
            )
        )

    return tuple(rows)


def run_report(module: str, timeout: float) -> Report:
    print(f"Running {module} ...", flush=True)
    started = perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = perf_counter() - started

    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"{module} exited with status {completed.returncode}.")

    rows = parse_rows(completed.stdout)
    if not rows:
        print(completed.stdout, end="")
        raise RuntimeError(f"{module} did not print any completed timing rows.")

    print(f"  ok: {len(rows)} rows in {elapsed:.3f}s", flush=True)
    return Report(module=module, elapsed=elapsed, rows=rows)


def load_baseline(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"reports": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(path: Path, reports: tuple[Report, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reports": {
            report.module: report.as_json()
            for report in reports
        }
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare_against_baseline(
    report: Report,
    baseline: dict[str, object],
    *,
    runtime_factor: float,
    step_factor: float,
    error_factor: float,
) -> list[str]:
    reports = baseline.get("reports")
    if not isinstance(reports, dict):
        return ["Baseline JSON must contain a 'reports' object."]

    baseline_report = reports.get(report.module)
    if not isinstance(baseline_report, dict):
        return []

    baseline_rows = baseline_report.get("rows")
    if not isinstance(baseline_rows, dict):
        return [f"{report.module}: baseline report has no rows object."]

    failures = []
    for row in report.rows:
        baseline_row = baseline_rows.get(row.key)
        if not isinstance(baseline_row, dict):
            failures.append(f"{report.module}: missing baseline row for {row.key}.")
            continue

        baseline_median = float(baseline_row["median"])
        if row.median > runtime_factor * baseline_median:
            failures.append(
                f"{report.module}: {row.key} median {row.median:.6g}s "
                f"exceeds {runtime_factor:g}x baseline {baseline_median:.6g}s."
            )

        baseline_error = float(baseline_row["error"])
        allowed_error = max(1.0e-15, error_factor * baseline_error)
        if row.error > allowed_error:
            failures.append(
                f"{report.module}: {row.key} error {row.error:.6g} "
                f"exceeds allowed {allowed_error:.6g}."
            )

        baseline_steps = baseline_row.get("steps")
        if row.steps is not None and baseline_steps is not None:
            allowed_steps = max(1, int(step_factor * int(baseline_steps)))
            if row.steps > allowed_steps:
                failures.append(
                    f"{report.module}: {row.key} steps {row.steps} "
                    f"exceeds allowed {allowed_steps}."
                )

    return failures


def selected_modules(names: tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        return REPORT_MODULES
    selected = []
    for name in names:
        if name in REPORT_MODULES:
            selected.append(name)
            continue
        module = f"competition.{name}.report"
        if module not in REPORT_MODULES:
            raise ValueError(f"Unknown competition report {name!r}.")
        selected.append(module)
    return tuple(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run competition reports with timeout and optional local baseline checks."
    )
    parser.add_argument("reports", nargs="*", help="Report names such as robertson or full module names.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-report timeout in seconds.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional JSON baseline to compare against.")
    parser.add_argument("--write-baseline", type=Path, default=None, help="Write current results to this JSON baseline.")
    parser.add_argument("--runtime-factor", type=float, default=2.0, help="Allowed median runtime multiplier.")
    parser.add_argument("--step-factor", type=float, default=1.2, help="Allowed accepted-step-count multiplier.")
    parser.add_argument("--error-factor", type=float, default=10.0, help="Allowed error multiplier.")
    args = parser.parse_args(argv)

    modules = selected_modules(tuple(args.reports))
    baseline = load_baseline(args.baseline)
    reports = []
    failures = []

    for module in modules:
        try:
            report = run_report(module, args.timeout)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            print(output, end="")
            failures.append(f"{module}: timed out after {args.timeout:g}s.")
            continue
        except Exception as exc:
            failures.append(str(exc))
            continue

        reports.append(report)
        failures.extend(
            compare_against_baseline(
                report,
                baseline,
                runtime_factor=args.runtime_factor,
                step_factor=args.step_factor,
                error_factor=args.error_factor,
            )
        )

    if args.write_baseline is not None and reports:
        write_baseline(args.write_baseline, tuple(reports))
        print(f"Wrote baseline: {args.write_baseline}")

    if failures:
        print()
        print("Competition check failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print()
    print("Competition check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
