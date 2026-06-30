# API reference

These pages are generated from docstrings. They are reference material, not the
main learning path. Start with the narrative docs unless you already know which
object you want to inspect.

The package reference is also a release smell detector. If this page exposes a
confusing helper, stale term, or accidental compatibility path, fix the source:
improve the docstring, internalise the object, or remove the object from the
public import surface.

## Package reference

This tree is generated from the `stark` package so docstrings from deeper
modules are visible in the built documentation.

```{toctree}
:maxdepth: 4

api/modules
```

## Curated surfaces

These pages group the main import surfaces by audience.

```{toctree}
:maxdepth: 2

stark
problem
methods
engines
diagnostics
core
```
