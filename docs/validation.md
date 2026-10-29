# Validation

The companion package is tested against independent small-system references
and between its two implementations. These tests establish the behaviour of
the released code; they are not a claim that one set of numerical parameters
converges every physical regime.

## Python

`pytest` checks:

- The spin algebra and physical basis convention.
- Stationary variance, lagged spatial covariance and proper complex noise for
  a bath with complex hopping and unequal losses.
- Unitary propagation of an undamped bath field.
- Isolated-system dynamics in both linear and nonlinear HOPS.
- Excitation transfer through two coupled modes against an explicit-cavity
  master equation. In this sector the checked excited-state density entries
  do not depend on the sampled Gaussian field.
- A nonlinear Rabi ensemble against the same independent master-equation
  implementation, using a fixed sampling tolerance.
- Timestep refinement with nested noise paths and larger local hierarchies.
- Ensemble connected correlations, partial traces and concurrence.
- Output-grid alignment at small and ordinary time scales, sparse output shapes,
  integer counts, prescribed-noise ownership and defensive model copies.

## Julia / ITensor

`Pkg.test()` checks the interleaved spin–auxiliary representation, compares
the full small-system MPO generator with a Python fixture, and evolves the
same prescribed noise path in both solvers. It also checks root normalisation,
compression diagnostics and the single-site case. The generator fixture
includes spin–spin interactions, bath hopping, unequal loss and nonlinear
coefficients. Its local hierarchy bounds differ between the two modes.

Compression regression tests independently inspect predictor truncation, compare
reported maxima with an explicit sequence of steps, and verify that sparse
observations retain the same cumulative diagnostics as observations at every step.
Parameter and time-grid failure cases are checked separately.

The fixtures use the Python convention in which the last tensor index varies
fastest. Julia tests explicitly map this ordering rather than relying on
reshape conventions. All fixture files are small plain CSVs. Run
`python tools/generate_fixtures.py --check` to compare the expected generator
and density against the current Python implementation. The script reads the
prescribed noise inputs without changing them; see [fixture provenance](../julia/test/fixtures/README.md).

## Release checks

The original 0.1.0 implementation was checked with Python 3.10, Julia 1.12.6,
ITensors 0.9.32 and ITensorMPS 0.4.2. The hardening pass adds public API
regressions, bringing the local suites to 35 Python tests and 44 Julia assertions
on those interpreter/package families. Both the Python physics tests and the
fixture-generation check also pass against an installed wheel built from the
source distribution.

GitHub Actions checks Python 3.10 and 3.12, a separate job at the declared
NumPy/SciPy/Matplotlib/pytest lower bounds, and Julia 1.10 and 1.12. It runs the
examples, verifies a representative archive replot and checks all data hashes.
The matrix and installed versions are visible in each workflow's logs.

## Relationship to the manuscript calculations

The released solvers are a small, independently tested implementation of the
same local hierarchy equations. They are not byte-identical copies of every
production solver. In particular, the specialised continuous-feedback solver
and full campaign scheduling are not included. The archived figure arrays
come directly from the production analysis and are not replaced by example
outputs from this package.
