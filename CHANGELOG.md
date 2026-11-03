# Changelog

## 0.1.0 — 3 November 2026

First public release.

- Linear and normalised nonlinear HOPS for arbitrary finite system matrices.
- Local cavity networks with nonuniform losses and correlated OU sampling.
- Object workflow through `Bath.sample`, `Model.solve` and ensemble observable
  methods on `Result`, with equivalent functional entry points.
- Julia/ITensor nonlinear spin-chain solver with separate auxiliary tensors,
  per-mode hierarchy bounds, compression diagnostics and a factorised MPO step.
  Julia also exposes `bath_kernel(engine, time)` and a standalone noise sampler.
- A two-spin example through time 24 with a fixed 4,096-trajectory ensemble per
  case, pointwise sampling intervals and independent numerical refinement checks.
- A two-qubit excitation-transport derivation, figures and benchmark data.
- A standalone correlated-noise study with a cumulative convergence animation,
  four static panels and covariance checks at multiple time origins.
- Input validation and numerical diagnostics, including cumulative Julia bond
  and discarded-weight extrema across predictor compression and unobserved steps.
- Independent master-equation checks, prescribed-noise cross-language fixtures,
  regression tests and CI checks of installed distributions and examples.
- Plotting inputs for all main and supplementary manuscript figures.
- Self-contained API and numerical-method guides, physical parameter records,
  software-version records, and separate software and numerical-data licences.
