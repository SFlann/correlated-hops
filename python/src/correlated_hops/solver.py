"""Dense-system, sparse-hierarchy linear and normalised nonlinear HOPS."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .bath import Bath, stationary_noise
from .hierarchy import hierarchy


@dataclass(frozen=True)
class Model:
    """H_S and F_i in H_int = sum_i (F_i a_i^dagger + F_i^dagger a_i)."""

    hamiltonian: np.ndarray
    couplings: tuple
    bath: Bath

    def __post_init__(self):
        H = np.array(self.hamiltonian, complex, copy=True)
        F = tuple(np.array(f, complex, copy=True) for f in self.couplings)
        if H.ndim != 2 or H.shape[0] != H.shape[1] or not len(H):
            raise ValueError("The system Hamiltonian must be square")
        if not np.allclose(H, H.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("The system Hamiltonian must be Hermitian")
        if len(F) != len(self.bath.h) or any(f.shape != H.shape for f in F):
            raise ValueError("Supply one system coupling matrix per bath mode")
        if not all(np.all(np.isfinite(a)) for a in (H, *F)):
            raise ValueError("System operators must be finite")
        for a in (H, *F):
            a.flags.writeable = False
        object.__setattr__(self, "hamiltonian", H)
        object.__setattr__(self, "couplings", F)


@dataclass
class Result:
    times: np.ndarray
    roots: np.ndarray  # [output time, trajectory, system basis]
    diagnostics: dict

    @property
    def density(self):
        """Average root projectors; linear roots are never renormalised here."""
        return (
            np.einsum("tsi,tsj->tij", self.roots, self.roots.conj())
            / self.roots.shape[1]
        )


def solve(
    model,
    initial,
    times,
    *,
    dt=0.01,
    kmax=4,
    samples=100,
    seed=0,
    nonlinear=True,
    noise=None,
):
    """Integrate on a uniform internal grid, with RK stages at noise half-steps.

    The fourth-degree RK update is applied to the prescribed interpolated
    noise path; this is not a claim of fourth-order stochastic convergence.
    Validate dt and kmax for the chosen observables. Normalising nonlinear
    roots includes the Girsanov shift and hierarchy response; normalising a
    linear trajectory alone would give the wrong estimator.
    """
    started = perf_counter()
    times = np.asarray(times, float)
    initial = np.asarray(initial, complex)
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must start at zero and increase strictly")
    if not np.all(np.isfinite(times)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid integration time grid")
    if int(samples) != samples or samples < 1:
        raise ValueError("samples must be a positive integer")
    steps = int(round(times[-1] / dt))
    outputs = np.rint(times / dt).astype(int)
    if not np.allclose(outputs * dt, times, atol=1e-11, rtol=1e-11):
        raise ValueError("Output times must lie on the internal grid")
    d = len(model.hamiltonian)
    m = len(model.couplings)
    if (
        initial.shape != (d,)
        or not np.all(np.isfinite(initial))
        or not np.isclose(np.linalg.norm(initial), 1)
    ):
        raise ValueError("initial must be a normalised finite system vector")
    bounds = np.full(m, kmax) if np.isscalar(kmax) else np.asarray(kmax)
    if (
        bounds.shape != (m,)
        or np.any(bounds < 1)
        or np.any(bounds != bounds.astype(int))
    ):
        raise ValueError("kmax must give a positive integer bound per mode")
    bounds = bounds.astype(int)
    base, annihilators, occupations = hierarchy(
        model.couplings, model.hamiltonian, model.bath.W, bounds
    )
    nh = len(occupations)
    z = (
        stationary_noise(model.bath, dt / 2, 2 * steps, samples, seed)
        if noise is None
        else np.asarray(noise, complex)
    )
    if z.shape != (2 * steps + 1, m, samples) or not np.all(np.isfinite(z)):
        raise ValueError(
            "noise must have shape (2*steps+1, modes, samples) and be finite"
        )
    state = np.zeros((nh * d, samples), complex)
    state[:d] = initial[:, None]
    shifts = np.zeros((m, samples), complex)

    def rhs(y, s, zz):
        a = base @ y
        yr = y.reshape(nh, d, samples)
        root = y[:d]
        norm = np.sum(abs(root) ** 2, axis=0)
        if nonlinear and np.any(norm < 1e-100):
            raise FloatingPointError(
                "The physical root vanished; reduce dt or inspect hierarchy stability"
            )
        beta = (
            np.array(
                [
                    np.einsum("ds,de,es->s", root.conj(), f.conj().T, root) / norm
                    for f in model.couplings
                ]
            )
            if nonlinear
            else np.zeros_like(s)
        )
        for i, f in enumerate(model.couplings):
            fy = np.einsum("de,hes->hds", f, yr).reshape(nh * d, samples)
            a += fy * (zz[i].conj() + (s[i] if nonlinear else 0))
            if nonlinear:
                a += (annihilators[i] @ y) * beta[i]
        ds = -model.bath.W.conj() @ s + beta if nonlinear else np.zeros_like(s)
        if nonlinear:
            gauge = np.real(np.sum(root.conj() * a[:d], axis=0)) / norm
            a -= y * gauge
        return a, ds

    roots = [state[:d].T.copy()]
    boundary = []
    next_output = 1
    max_norm_correction = 0.0
    for step in range(steps):
        a, sa = rhs(state, shifts, z[2 * step])
        b, sb = rhs(state + dt * a / 2, shifts + dt * sa / 2, z[2 * step + 1])
        c, sc = rhs(state + dt * b / 2, shifts + dt * sb / 2, z[2 * step + 1])
        e, se = rhs(state + dt * c, shifts + dt * sc, z[2 * step + 2])
        state += dt * (a + 2 * b + 2 * c + e) / 6
        shifts += dt * (sa + 2 * sb + 2 * sc + se) / 6
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(shifts)):
            raise FloatingPointError(
                "Nonfinite trajectory; no samples have been discarded"
            )
        if nonlinear:
            scale = np.sqrt(np.sum(abs(state[:d]) ** 2, axis=0))
            if np.any(scale < 1e-100):
                raise FloatingPointError("Vanishing root norm")
            max_norm_correction = max(
                max_norm_correction, float(np.max(abs(scale - 1)))
            )
            state /= scale
        if next_output < len(outputs) and step + 1 == outputs[next_output]:
            roots.append(state[:d].T.copy())
            next_output += 1
            weights = np.sum(abs(state.reshape(nh, d, samples)) ** 2, axis=1)
            boundary.append(
                [
                    float(
                        np.mean(weights[occupations[:, i] == k].sum(0) / weights.sum(0))
                    )
                    for i, k in enumerate(bounds)
                ]
            )
    return Result(
        times.copy(),
        np.asarray(roots),
        dict(
            method="nonlinear HOPS" if nonlinear else "linear HOPS",
            samples=samples,
            dt=dt,
            kmax=bounds.tolist(),
            hierarchy_states=nh,
            seed=seed,
            elapsed_seconds=perf_counter() - started,
            max_root_norm_correction=max_norm_correction,
            auxiliary_boundary_fractions=boundary,
            boundary_note="Diagnostic hierarchy weights, not photon populations or error bounds",
        ),
    )
