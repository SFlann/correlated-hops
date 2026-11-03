# Example data and validation records

This archive contains the numerical data for the documentation's models and
sampling calculations. It is separate from the manuscript figure archive in
`data/figures`. Arrays are [CC BY 4.0](../LICENSE); code is MIT licensed.

| Data set | Model and contents |
| --- | --- |
| [JC–Rabi transport](earlier_note/) | Resonant two-qubit JC–Rabi comparison: reduced densities, photon and mixed moments, batch uncertainty, and numerical checks through times 12 and 24. |
| [Normal-mode noise](earlier_noise/) | Ten-mode free bath: 100,000-history normal-mode noise calculation, complex covariance checks, and photon-propagation controls. |
| [Short-time two-spin control](original_readme/) | Detuned two-spin model through time 2: example source, parameters and a fixed 2,048-trajectory sampling check. |
| [Local-basis noise](package_noise/) | Ten-mode free bath: local-basis `Bath.sample` calculation, complex estimates, seed scheme and validation. |
| [Two-spin dynamics](package_pair/) | Detuned two-spin model through time 24: ensemble densities, uncertainty intervals and timestep, hierarchy and reference-cutoff checks. |

The [transport guide](../../docs/two-qubit-note.md),
[noise guide](../../docs/noise.md) and [validation guide](../../docs/validation.md)
state the physical models, numerical methods and estimator conventions.
The transport benchmark includes physical photon estimators beyond the
reduced-system `Result` API.

`manifest.json` records repository-relative asset paths, sizes and SHA-256
hashes. Its `source` fields are provenance identifiers, not paths required
at runtime. Every referenced asset is included in the repository. Verify them
from the repository root with `python tools/verify_example_assets.py`.

Simulation scripts write to `examples/output` by default. Saved complex arrays
and explicit settings support independent inspection of the plotted values
and preserve the distinction between physical parameters, numerical settings
and stochastic samples.
