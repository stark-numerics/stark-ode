# Tests Design Notes

The test suite should verify public behaviour, numerical accuracy, and the hot
path promises that matter to users and contributors. It should not freeze old
internal shapes just because they once existed.

## Shared Test Machinery

Prefer the helpers in `tests.support` before writing a local dummy class.

Those helpers are deliberately small, typed implementations of the same
contracts that user code is expected to satisfy. Reusing them keeps tests
focused on the behaviour under test instead of on one-off local fakes. It also
makes Pyright output more useful: when a shared helper satisfies the contract,
remaining type errors are much more likely to point at the package code or at a
real test mismatch.

Good shared helpers are not elaborate fixtures. They should be tiny concrete
objects with meaningful docstrings, such as `DummyScalarTranslation`,
`DummyArrayTranslation`, `DummyVectorTranslation`, `DummyVectorBasis`,
`DummyStructuredTranslation`, and `DummyTableauSpecialist`.
Hover text in an IDE is part of the documentation layer, so avoid say-what-you-
see docstrings and explain why the object exists.

## Naming

Use `Dummy...` for reusable test implementations and for local test objects that
stand in for a real package concept. Avoid `Fake...` and `...Fixture`; those
names have drifted in meaning and make it harder to see whether an object is a
contract-shaped test implementation or pytest setup machinery.

Local helpers are still fine when a test needs deliberately unusual behaviour,
for example a translation basis that returns a fresh vector instead of mutating
the supplied output. In that case the helper should be named for the behaviour
being tested and should have a docstring explaining the oddity.

## Coverage Shape

Tests should usually sit near the package area they protect. The suite does not
need repeated accuracy tests for the same scheme in several places unless each
test covers a distinct user-visible path. When cleaning the suite for release,
delete tests that only preserve private implementation details, and rewrite
tests that describe real behaviour through stale internals.
