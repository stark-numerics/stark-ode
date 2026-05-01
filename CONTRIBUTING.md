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

Check pytest discovery after moving tests with:

```powershell
python -m pytest --collect-only
```

Focused package-area commands:

```powershell
python -m pytest tests/interface
python -m pytest tests/carriers
python -m pytest tests/algebraist
python -m pytest tests/schemes
python -m pytest tests/resolvents tests/inverters
python -m pytest tests/comparison
```

Run slow tests with:

```powershell
python -m pytest -m slow
```

Run slow integration tests with:

```powershell
python -m pytest tests/integration -m slow
```

Examples and benchmarks are not part of the pytest suite. Run them directly
when changing example or benchmark code.

## Code style

STARK has a project-specific house style. Please read:

- [`docs/house_style.md`](docs/house_style.md)

That document is the primary reference for naming, architecture, callable
classes, hot-path structure, and the package's numerical-programming style.
