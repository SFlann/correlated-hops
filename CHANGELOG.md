# Changelog

## Unreleased

- Add the compatible object workflow `Bath.sample`, `Model.solve` and ensemble
  observable methods on `Result`; expose Julia's `bath_kernel(engine, time)`.
- Extend the original README physics example through time 24 with a fixed
  4,096-trajectory ensemble per case and independent numerical refinement checks.
- Include the earlier two-qubit note, its original figures and data provenance.
- Add a standalone correlated-noise study, cumulative convergence animation,
  static panels and tests of covariance at multiple time origins.

- Reject output times that map to the same integration step and validate integer
  counts consistently across Python entry points.
- Include predictor compression and all unobserved steps in Julia simulation
  diagnostics, retaining cumulative bond and discarded-weight extrema.
- Add API contracts, regression tests, reproducible cross-language fixture
  tooling, release checks and tests of installed distributions.
- Label example methods and pointwise sampling uncertainty; record physical
  parameters and software versions alongside the example output.
- Improve source formatting, source-distribution contents and licence metadata.

## 0.1.0 — 29 October 2026

- Linear and normalised nonlinear HOPS for arbitrary finite system matrices.
- Local cavity networks with nonuniform losses and correlated OU sampling.
- Julia/ITensor nonlinear spin-chain solver with separate auxiliary tensors,
  per-mode hierarchy bounds, compression diagnostics and a factorised MPO step.
- Independent small-system reference calculations and cross-language tests.
- Plotting inputs for all main and supplementary manuscript figures.
