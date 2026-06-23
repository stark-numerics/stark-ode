# Contributing

Thanks for working on STARK.

## Setup

Install the development dependencies with:

```powershell
python -m pip install -e ".[dev]"
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

## See also

- [`docs/contributing/README.md`](docs/contributing/README.md)

