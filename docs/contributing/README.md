# Contributor notes

This directory is for contribution workflow and house style. It is not the
main user manual.

## Start here

- [House style](house-style.md): naming, layering, and example-writing conventions.

## Local design notes

Implementation packages may contain a local `DESIGN.md`. These files explain
why a package is shaped the way it is, which paths are hot, and which extension
points are deliberate. They are meant to preserve design intent for
contributors and future maintenance agents.

Current local design notes include:

- `stark/core/contracts/DESIGN.md`
- `stark/engines/generator/DESIGN.md`
- `stark/engines/allocator/DESIGN.md`
- `stark/methods/schemes/DESIGN.md`
- `stark/methods/resolvents/DESIGN.md`
- `stark/methods/inverters/DESIGN.md`

When changing a deep subsystem, read the nearest `DESIGN.md` before patching.
If the code has moved away from the note, update the note in the same change.

## Benchmarking and performance notes

Benchmarking guidance should live with the benchmark tools.
User-facing timing interpretation belongs in `docs/diagnostics.md`.
