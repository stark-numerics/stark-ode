# Frame Design Notes

The frame package is moving toward typing by protocol rather than by concrete
classes.

`Frame` remains the user-facing way to declare structured state in `stark-ode`,
and `Field` remains the default concrete field implementation. However, frame
contents are intended to be accepted structurally through contracts such as
`FieldLike`, `NormLike`, and `InnerProductNamed`.

This matters because downstream packages need to enrich the same concepts
without fighting concrete type checks. For example, `stark-pde` may need a
PDE-specific field that carries spatial lattice, boundary, plotting, or domain
metadata while still satisfying the core STARK field contract consumed by
allocation and algebra machinery.

The desired direction is:

- concrete objects provide convenient defaults for ordinary `stark-ode` users
- core contracts describe the behavior engines require
- frame and algebraist code depend on those contracts, not exact classes
- algebraist-specific views, such as included norm entries, are derived by
  algebraist helpers rather than required on the frame contract
- specialized packages can provide enriched fields or frames that duck type into
  the same machinery

This is intentionally a gradual migration. Compatibility constructors such as
string paths and mapping specs should keep producing ordinary `Field` objects, while internal storage and engine-facing APIs should avoid unnecessary concrete `Field` requirements.
