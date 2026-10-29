# Numerical conventions

For system operators F_i, the interaction is
H_int = sum_i (F_i a_i^dagger + F_i^dagger a_i). The bath Hamiltonian h is
Hermitian; independent vacuum output ports have photon-number loss rates kappa_i.

The drift is W = diag(kappa_i/2) + i h and the stationary vacuum kernel is
K(t) = exp(-W t), t >= 0. The Gaussian grid sampler uses the exact transition
P = exp(-W dt) and covariance Q = I - P P^dagger. It therefore retains spatial
covariance when h and the damping matrix do not commute.

In the factorially scaled hierarchy, the linear generator contains

```
-i H_S + sum_i F_i z_i* + sum_i (F_i b_i^dagger - F_i^dagger b_i)
       - sum_ij W_ij b_i^dagger b_j.
```

The root has all auxiliary indices zero. Nonlinear HOPS adds the Girsanov shift
s_i to z_i*, adds beta_i b_i with beta_i = <F_i^dagger> in the normalised root,
and evolves ds/dt = -W* s + beta. A scalar normalisation gauge is applied to
the whole hierarchy. Both the shifts and response terms matter: simply
normalising the linear roots is not nonlinear HOPS.

Python uses sparse hierarchy matrices and dense system vectors. Its RK stages
sample a prescribed coloured-noise path at endpoints and midpoints. Julia
uses a midpoint predictor followed by four factors of the degree-four Taylor
polynomial of a frozen MPO generator. Neither implementation claims global
fourth-order accuracy for stochastic, nonlinear evolution. Time-step checks
must use nested samples of the same underlying noise path.

Local hierarchy bounds can differ between modes. The Julia solver additionally
uses an SVD discarded-weight cutoff and a maximum MPS bond dimension. Boundary
weights, discarded weights and whether the bond cap was reached are diagnostics;
none alone certifies observable convergence. Check observables with larger
hierarchies, smaller timesteps and tighter MPS parameters when using new regimes.

For nonlinear HOPS, average normalised root projectors to obtain the density.
For linear HOPS, average unnormalised root projectors without self-normalising
the ensemble. For connected correlations, first average <AB>, <A> and <B>, then
form <AB> - <A><B>. For concurrence, first construct the averaged two-spin density.
The auxiliary norm is not a physical photon population.

The Julia representation uses dense local spin tensors. It does not claim a
conserved U(1) quantum number for Rabi or transverse couplings, which break that
symmetry. Fermionic models and their Jordan--Wigner strings are not included.

## Background

- D. Süß, A. Eisfeld and W. T. Strunz, *Hierarchy of Stochastic Pure States for
  Open Quantum System Dynamics*, Physical Review Letters **113**, 150403 (2014).
  https://doi.org/10.1103/PhysRevLett.113.150403
- R. Hartmann and W. T. Strunz, *Exact Open Quantum System Dynamics Using the
  Hierarchy of Pure States (HOPS)*, Journal of Chemical Theory and Computation
  **13**, 5834–5845 (2017). https://doi.org/10.1021/acs.jctc.7b00751
- S. Flannigan, F. Damanet and A. J. Daley, *Many-Body Quantum State Diffusion
  for Non-Markovian Dynamics in Strongly Interacting Systems*, Physical Review
  Letters **128**, 063601 (2022). https://doi.org/10.1103/PhysRevLett.128.063601
