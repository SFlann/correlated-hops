# Example and earlier-study records

This archive is separate from the immutable manuscript figure archive in
`data/figures`. Arrays are CC BY 4.0, as specified in `data/LICENSE`.
Code remains MIT licensed. The original authored note and its figures are
included with their source provenance.

- `earlier_note`: original September JC–Rabi arrays and numerical records.
- `earlier_noise`: original 100,000-history animation data and propagation controls.
- `original_readme`: the superseded short-time example, parameters and fixed
  2,048-trajectory sampling check; its original image is retained in `docs/images`.
- `package_noise`: new standalone `Bath.sample` calculation with the original
  ten-mode physical parameters and a separately recorded seed scheme.
- `package_pair`: the extended original README model and numerical qualification.

`manifest.json` records source paths, sizes and SHA-256 hashes. Run
`python tools/verify_example_assets.py` from the repository root. New simulations
write to `examples/output`, never to this archive by default. The new examples
store complex densities/estimates and explicit settings, so plots can be audited
without treating images as numerical data.
