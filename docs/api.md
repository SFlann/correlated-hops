# API guide

All energies are angular frequencies, with hbar = 1. A loss rate `kappa`
describes photon **number** decay; field amplitudes decay at `kappa/2`.
Spin matrices use the local basis `|down>, |up>`; the rightmost tensor index
varies fastest in Python. See [the method](method.md) for the equations.

## Python

The object workflow is `Bath` → `Model` → `Result`. A model owns its physical
operators and bath; a result owns its output times, trajectory roots and
diagnostics. The numerical routines have a single implementation: the
established `solve(model, ...)`, `stationary_noise(bath, ...)` and observable
functions remain supported alongside these convenience methods.

```python
from correlated_hops import spin_chain, spin_operators

model = spin_chain(2, g=0.3, Jb=0.25, kappa=[0.1, 0.4])
result = model.solve([1, 0, 0, 0], [0, 0.5, 1.0], samples=256)
sz = spin_operators(2)["sz"]
mean = result.expectation(sz[0])
correlation = result.connected(*sz)
entanglement = result.concurrence()
first_spin = result.partial_trace([0], [2, 2])
```

`Bath(h, kappa)` accepts a Hermitian bath matrix of shape `(m, m)` and
nonnegative losses of shape `(m,)`. `Model(hamiltonian, couplings, bath)`
accepts a Hermitian system matrix of shape `(d, d)` and one coupling matrix
of that shape per bath mode. Both classes copy their arrays and mark those
copies read-only. Mutating the caller's input arrays does not alter the model.

`spin_chain(n, ...)` constructs a dense small-system model; `spin_operators(n)`
returns a dictionary of lists of dense site operators. The available keys are
`sm`, `sp`, `sx`, `sy` and `sz`. Coupling choices are `rabi`, `sx` and `sz`;
the builder's defaults and interaction convention are documented in its docstring.

```python
result = model.solve(
    initial, times, dt=0.01, kmax=[3, 4], samples=100, seed=17, nonlinear=True
)
```

- `initial` is a finite, normalised complex system vector of shape `(d,)`.
- `times` is a finite one-dimensional array starting at zero, with at least
  two strictly increasing entries on distinct internal grid steps. Each time
  divided by `dt` must be within `1e-9` of an integer. Outputs can skip steps.
- `kmax` is a positive integer or a length-`m` sequence of positive integers.
  It is the maximum hierarchy index; each local dimension is `kmax + 1`.
- Counts accept Python and NumPy integers. Floats and booleans are rejected.
- `noise=None` generates stationary complex noise using `seed`.
  Alternatively pass a finite complex array of shape
  `(2 * steps + 1, m, samples)`, where `steps = round(times[-1] / dt)`.
  Noise samples are at half-step intervals. Supplied noise is read without
  modification, and the seed does not affect that trajectory.

For timestep comparisons, generate one fine noise path and subsample it.
The following compares the same path at two resolutions:

```python
from correlated_hops import stationary_noise, solve

z = stationary_noise(model.bath, 0.0025, 160, 2, seed=23)
fine = solve(model, initial, [0, 0.2, 0.4], dt=0.005, samples=2, noise=z)
coarse = solve(model, initial, [0, 0.2, 0.4], dt=0.01, samples=2, noise=z[::2])
```

`bath.sample(dt, steps, samples, seed=0)` and the equivalent
`stationary_noise(bath, dt, steps, samples, seed=0)` include the initial
sample, so its shape is `(steps + 1, m, samples)`. Zero steps are supported.
Here `dt` is the spacing of the noise samples themselves. `bath.kernel(time)`
returns the full complex matrix at a finite nonnegative lag. Both methods work
without a system model; see the [standalone noise study](noise.md).

### Results and diagnostics

| Field | Meaning |
| --- | --- |
| `result.times` | Copy of the requested output times |
| `result.roots` | Complex array of shape `(len(times), samples, d)` |
| `result.density` | Ensemble root projectors, shape `(len(times), d, d)`; computed on access |
| `result.diagnostics` | Method, sample count, timestep, hierarchy bounds, seed, noise source and runtime |
| `max_root_norm_correction` | Largest nonlinear root-norm correction across all steps and samples |
| `auxiliary_boundary_fractions` | Mean boundary weight per mode at output times after the initial state |

Nonlinear roots are normalised. Linear roots remain unnormalised, and their
ensemble density is **not** divided by its trace. With a finite ensemble that
trace can differ from one. `noise_source` distinguishes generated and prescribed
paths; for prescribed noise retain the input array as part of your own run record.

`result.expectation(operator)`, `result.connected(first, second)`,
`result.partial_trace(keep, dimensions)` and `result.concurrence()` evaluate
the corresponding functions on the ensemble density. They retain the linear
solver's unnormalised-estimator convention; concurrence still requires a
normalised positive density.

`expectation`, `connected`, `partial_trace` and `concurrence` accept leading
time/batch axes on the density. Evaluate connected correlations and concurrence
from the ensemble density, after averaging the stochastic projectors.
`partial_trace(density, keep, dimensions)` preserves subsystem order and uses
zero-based increasing `keep` indices. Concurrence requires a positive,
normalised two-qubit density.

`reference.master_equation(model, initial, times, photon_cutoff=3)` returns
reduced system densities on an increasing, equally spaced time grid starting
at zero. Each explicit cavity has dimension `photon_cutoff + 1`. Check this
reference's own cutoff convergence.

Invalid model parameters, grids, counts and prescribed-noise shapes raise
`ValueError`. Vanishing/nonfinite solver trajectories raise `FloatingPointError`;
failed samples are never silently removed.

## Julia / ITensor

`Engine(n; ...)` constructs an interleaved `spin1, aux1, spin2, aux2, ...`
representation. `kappa` and `g` can be scalars or length-`n` vectors;
`kmax` can be one integer or a length-`n` integer vector.
`coupling` is `:rabi`, `:sx` or `:sz`; supply a Hermitian `h` to replace
the default nearest-neighbour bath Hamiltonian.

`initial_state(engine; preparation=:neel)` supports `:down`, `:neel`
and `:x_neel`. `root_mps(engine, psi; normalize=true)` returns the projected
physical MPS and its squared norm **before** normalisation.
`means(engine, psi)` returns the normalised root expectations of the adjoint
coupling operators, one per bath mode.

`simulate(engine; T=1.0, dt=0.01, seed=0, preparation=:neel,
observe_every=10, callback=nothing)` evolves **one trajectory**, returning:

- `state`: final spin–hierarchy MPS;
- `shift`: final complex Girsanov shift vector;
- `records`: observations at zero, each observation interval, and the final time.

Each record contains `time`, `moments` and `diagnostics`. The callback,
when supplied, receives these three values. Moments contain length-`n`
`singles_first` and `singles_second` vectors and an `n × n` `pairs` matrix.
The default observable is `Sz`; use `spin_correlations(engine, psi;
first="Sx", second="Sy")` for other spin moments.

Average each raw moment over independent trajectories before forming
connected correlators. A single simulated trajectory is not an ensemble estimate.

### Compression diagnostics

In `simulate` records:

- `maximum_bond_dimension` is the largest retained bond dimension since the start;
- `bond_cap_reached` remains true once any predictor or polynomial factor reaches the cap;
- `largest_discarded_weight` is the largest reported per-bond discarded weight
  since the start, including the predictor and unobserved steps;
- `bond_dimensions` and `auxiliary_boundary_fractions` describe the current state.

The SVD `cutoff` is a discarded-weight tolerance, not an elementwise tensor
threshold. `maxdim` imposes an additional cap. These quantities are diagnostics,
not observable error bounds or a replacement for convergence comparisons.

For a prescribed path, `advance(engine, psi, shift, z0, zm, dt)` returns
`(new_state, new_shift, discarded_weights)`. The coefficient vectors have
length `n`; `z0` and `zm` are the noise at the step's start and midpoint.
The inputs are read without modification. The discarded weights include
the predictor and all four polynomial factors. Calling `diagnostics(engine,
psi, discarded_weights)` directly reports current bond/boundary snapshots
and the maximum of the supplied weight history.

`bath_kernel(engine, time)` returns the complex local-bath kernel at finite
nonnegative lag, with system coupling strengths kept separate.

Julia's `stationary_noise(engine, dt, steps, seed=0)` returns an array of
shape `(n, steps + 1)` for one trajectory. Python and Julia RNGs do not
produce the same path from the same seed; cross-language comparisons must
use a prescribed shared path. Invalid parameters raise `ArgumentError`;
numerical failures raise an error instead of discarding a trajectory.
