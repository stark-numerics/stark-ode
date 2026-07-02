# Contributing

Thanks for working on STARK.

## Setup

Install the development dependencies with:

```powershell
python -m pip install -e ".[dev]"
```

Install the documentation dependencies with:

```powershell
python -m pip install -e ".[docs]"
```

## Tests

Run the default test suite with:

```powershell
python -m pytest
```

Check pytest discovery after moving tests with:

```powershell
python -m pytest --collect-only
```

Examples and benchmarks are not part of the pytest suite. Run them directly
when changing example or benchmark code.

## Documentation

The narrative docs live in `docs/` and are written in Markdown. Generated API
reference docs should support the narrative docs, not replace them.

Build the docs with Sphinx:

```powershell
sphinx-build -b html docs docs/_build/html
```

Use generated docs as a smell detector: if an API page is confusing, the
public name, import path, signature, or docstring probably needs attention.

Contributor-facing architecture notes live in local `DESIGN.md` files inside
the package. Start with `docs/contributing/README.md` when changing package
structure.

## See also

- [`docs/contributing/README.md`](docs/contributing/README.md)
