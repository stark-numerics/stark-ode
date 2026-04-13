# Contributing

Thanks for working on STARK.

## Setup

Install the development dependencies with:

```powershell
python -m pip install -e ".[dev]"
```

For example and notebook work:

```powershell
python -m pip install -e ".[examples,notebooks]"
```

For benchmark work:

```powershell
python -m pip install -e ".[benchmarks]"
```

## Tests

Run the default test suite with:

```powershell
python -m pytest
```

Run slow tests with:

```powershell
python -m pytest -m slow
```

## Code style

STARK has a project-specific house style. Please read:

- [`docs/house_style.md`](docs/house_style.md)

That document is the primary reference for naming, architecture, callable
classes, hot-path structure, and the package's numerical-programming style.
