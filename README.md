# Correlated HOPS

**Stochastic pure-state dynamics for systems coupled to communicating environments.**

A compact research companion to *Correlations carried and controlled by coupled
non-Markovian environments*, by Stuart Flannigan. The physical system is a set of
spins; local auxiliary hierarchies retain the memory and transport of eliminated
cavity modes.

The repository provides:

- A Python implementation of linear and normalised nonlinear HOPS for small systems.
- A Julia/ITensor MPS implementation of nonlinear HOPS for spin chains.
- The numerical data behind the manuscript figures, including reference curves
  and the displayed uncertainty intervals.
- Short examples and tests against explicit-cavity master equations and across
  the two implementations.

The solvers use the **local bath basis**. Photon hopping enters both the coloured
noise and the hierarchy drift, including when neighbouring cavities have unequal
losses. No normal-mode approximation to the damping matrix is made.

## Python: two locally coupled spins

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot,test]'
python examples/python/rabi_pair.py
pytest
```

```python
import numpy as np
from correlated_hops import spin_chain, solve, spin_operators, connected

model = spin_chain(2, g=0.3, Jb=0.25, kappa=[0.1, 0.4])
result = solve(model, [1, 0, 0, 0], np.linspace(0, 1, 11),
               dt=0.01, kmax=[3, 4], samples=100, seed=17)
sz = spin_operators(2)['sz']
correlation = connected(result.density, sz[0], sz[1]).real
```

`kappa` is the photon-**number** loss rate; amplitude damping is `kappa/2`.
All energies are angular frequencies and hbar is one. Each `kmax` is an upper
hierarchy index, so the associated local dimension is `kmax + 1`.

## Julia: tensor-network dynamics

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=julia examples/julia/spin_chain.jl
```

The MPS alternates physical spins and local hierarchy tensors. The example uses
a short XXZ chain; changing the chain length does not require constructing its
dense Hilbert space. The integrator applies four factorised polynomial MPOs
after a midpoint predictor. See [the numerical conventions](docs/method.md)
for coefficient updates, truncation and observable estimation.

## Scope

Version 0.1 implements time-independent spin Hamiltonians coupled to passive,
zero-temperature Gaussian cavity networks, with initially factorised vacuum
environments. Python accepts arbitrary finite system matrices. The Julia API
provides Rabi, transverse and longitudinal local couplings and XXZ interactions.

The manuscript also studies monitored output and feedback. Its feedback figure
data are included, but the specialised feedback production solver is outside
this small companion package. The package is not a replacement for the full
simulation campaign, and its examples are not production convergence settings.

## Data and reproducibility

See [the data guide](data/README.md), [validation](docs/validation.md), and
[contribution guidance](CONTRIBUTING.md). Figure data are kept separately from
solver examples: installing or testing the package never changes manuscript data.
Connected correlations and concurrence are evaluated from the **ensemble density**,
not by averaging their nonlinear values on individual stochastic states.

## Citation and licence

Use `CITATION.cff` to cite this software. The manuscript citation can be added
when its public identifier is available. Software is MIT licensed; numerical
data are CC BY 4.0. See `LICENSE` and `data/LICENSE`. Schematic artwork is not
part of the numerical-data release.
