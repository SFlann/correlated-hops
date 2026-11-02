# Correlated HOPS

[![Tests](https://github.com/SFlann/correlated-hops/actions/workflows/tests.yml/badge.svg)](https://github.com/SFlann/correlated-hops/actions/workflows/tests.yml)

**Stochastic pure-state dynamics for systems coupled to communicating environments.**

A compact research companion to *Correlations carried and controlled by coupled
non-Markovian environments*, by Stuart Flannigan. The physical system is a set of
spins; local auxiliary hierarchies retain the memory and transport of eliminated
cavity modes.

Use it to explore how spins become correlated when their local environments
exchange excitations and retain memory. The package is designed as a small,
readable reference implementation with independent numerical checks.

The repository provides:

- A Python implementation of linear and normalised nonlinear HOPS for small systems.
- A Julia/ITensor MPS implementation of nonlinear HOPS for spin chains.
- A standalone correlated Gaussian-noise sampler and an animated convergence demo.
- The earlier two-qubit note, its plots and numerical records.
- The numerical data behind the manuscript figures, including reference curves
  and the displayed uncertainty intervals.
- Short examples and tests against explicit-cavity master equations and across
  the two implementations.

The solvers use the **local bath basis**. Photon hopping enters both the coloured
noise and the hierarchy drift, including when neighbouring cavities have unequal
losses. No normal-mode approximation to the damping matrix is made.

## Python: two locally coupled spins

```sh
git clone https://github.com/SFlann/correlated-hops.git
cd correlated-hops
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot,test]'
python examples/python/rabi_pair.py
pytest
```

Two initially uncorrelated spins develop correlations and entanglement through
their communicating environments, with no direct spin–spin interaction. This
extends the original example to $\omega_c t=24$ at the **same physical
parameters**, comparing hopping $J_b=0.25$ with the uncoupled-bath control $J_b=0$.

The figure uses 4,096 trajectories per case. Solid curves are nonlinear HOPS,
shading is a pointwise 95% interval from resampling independent trajectory
batches, and dotted curves are independent finite-cutoff master equations.
Connected correlations and concurrence are formed from the ensemble density.
The script writes the plot, numerical arrays and a JSON record to
`examples/output/`. Add `--validate` to check nested-noise timestep refinement,
larger hierarchies and the reference's photon cutoff. See the
[numerical record](data/examples/package_pair/rabi_pair.json).

![Two-spin example with sampling uncertainty and independent references](docs/images/rabi_pair.png)

To replot the checked data without rerunning the ensemble:

```sh
python examples/python/rabi_pair.py --input-data data/examples/package_pair/rabi_pair.npz
```

A small exploratory run through the object API:

```python
import numpy as np
from correlated_hops import spin_chain, spin_operators

model = spin_chain(2, g=0.3, Jb=0.25, kappa=[0.1, 0.4])
result = model.solve(
    [1, 0, 0, 0],
    np.linspace(0, 1, 11),
    dt=0.01,
    kmax=[3, 4],
    samples=100,
    seed=17,
)
sz = spin_operators(2)["sz"]
correlation = result.connected(sz[0], sz[1]).real
entanglement = result.concurrence()
```

`kappa` is the photon-**number** loss rate; amplitude damping is `kappa/2`.
All energies are angular frequencies and hbar is one. Each `kmax` is an upper
hierarchy index, so the associated local dimension is `kmax + 1`.

## Excitation transfer and return: the earlier note

The [two-qubit note and figure gallery](docs/two-qubit-note.md) follow an
excitation from the first qubit into its cavity, through the hopping link,
into the second qubit, and back again. The JC and full-Rabi Hamiltonians give
different transfer, return and entanglement dynamics.

![Original JC and Rabi excitation-transfer and return plots](docs/images/note_excitation_flow.png)

These are the original note's independently validated calculations, with their
original 4,096-trajectory ensembles and 95% batch intervals. The
[PDF](docs/studies/correlated_hops_note.pdf),
[JC–Rabi comparison](docs/images/note_jc_vs_rabi.png),
[photon observables](docs/images/note_photon_observables.png), and
[qubit–photon correlations](docs/images/note_system_mode_correlations.png)
are included with [numerical data and provenance](data/examples/README.md).
The gallery explains the earlier solver's additional bath-observable estimators.

## Correlated noise: watch the kernel emerge

The same `Bath` object can generate spatially and temporally correlated complex
Gaussian fields independently of any spin simulation. It preserves the full
matrix covariance, including nonuniform cavity losses.

![Increasing the ensemble from 10 to 100000 noise trajectories](docs/images/noise_convergence.gif)

The animation uses the public package sampler on the earlier study's ten-mode
hopping chain. Each frame adds trajectories to the same cumulative ensemble:
the individual paths remain random while their complex correlations converge.
The displayed magnitude is taken **after** averaging the complex products.

[Four-panel static version](docs/images/noise_convergence_panels.png) ·
[Original 34-second video](docs/media/noise_convergence_original.mp4) ·
[Derivation, propagation plots and checks](docs/noise.md)

```python
import numpy as np
from correlated_hops import Bath

bath = Bath([[1.0, 0.25], [0.25, 1.0]], kappa=[0.1, 0.4])
z = bath.sample(dt=0.1, steps=200, samples=2000, seed=17)
# Shape: (time, mode, independent trajectory)
sampled = (z * z[0, 0, :].conj()).mean(axis=-1)
exact = np.array([bath.kernel(t)[:, 0] for t in np.arange(201) * 0.1])
```

Run `python examples/python/noise_convergence.py` to generate the GIF, four-panel
figure, complex estimates and validation record. Sampling and rendering use the
standard plotting dependencies; no external video encoder is needed.

## Julia: tensor-network dynamics

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=julia examples/julia/spin_chain.jl
```

Julia uses an `Engine` object with ordinary multiple-dispatch methods such as
`simulate(engine, ...)`, `bath_kernel(engine, time)` and
`stationary_noise(engine, ...)`. The MPS alternates physical spins and local
hierarchy tensors. The example uses
a short XXZ chain; changing the chain length does not require constructing its
dense Hilbert space. The integrator applies four factorised polynomial MPOs
after a midpoint predictor. See [the numerical conventions](docs/method.md)
for coefficient updates, truncation and observable estimation.

## Design

- **Dense Python baseline:** sparse hierarchy matrices and dense system vectors
  keep small-system calculations straightforward to inspect and validate.
- **Julia tensor representation:** separate spin and hierarchy tensors permit
  chain calculations without constructing the dense system Hilbert space.
- **Small object API:** `Bath.sample`, `Model.solve` and ensemble observables on
  `Result`; the original function API remains compatible.
- **Independent checks:** explicit-cavity master equations test the numerical
  dynamics; prescribed-noise fixtures compare the two implementations.
- **Separate data archive:** examples write their own outputs. The manuscript
  inputs remain immutable and are checked against a SHA-256 manifest.

## Scope

The package implements time-independent spin Hamiltonians coupled to passive,
zero-temperature Gaussian cavity networks, with initially factorised vacuum
environments. Python accepts arbitrary finite system matrices. The Julia API
provides Rabi, transverse and longitudinal local couplings and XXZ interactions.

The manuscript also studies monitored output and feedback. Its feedback figure
data are included, but the specialised feedback production solver is outside
this small companion package. The package is not a replacement for the full
simulation campaign, and its examples are not production convergence settings.

## Data and reproducibility

See the [API guide](docs/api.md), [data guide](data/README.md),
[validation](docs/validation.md), [contribution guidance](CONTRIBUTING.md) and
[release checks](docs/releasing.md). Figure data are kept separately from
solver examples: installing or testing the package never changes manuscript data.
Connected correlations and concurrence are evaluated from the **ensemble density**,
not by averaging their nonlinear values on individual stochastic states.

## Citation and licence

Use `CITATION.cff` to cite this software. The manuscript citation can be added
when its public identifier is available. Software is MIT licensed; numerical
data are CC BY 4.0. See `LICENSE` and `data/LICENSE`. Schematic artwork is not
part of the numerical-data release.
