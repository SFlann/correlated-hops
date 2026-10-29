"""Passive vacuum bath and exact stationary Ornstein--Uhlenbeck sampling."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


def positive_factor(covariance):
    """Square root of a positive semidefinite Hermitian covariance."""
    covariance = (covariance + covariance.conj().T) / 2
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-11 * max(1.0, np.linalg.norm(covariance)):
        raise ValueError("The noise covariance is not positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0))[None, :]


@dataclass(frozen=True)
class Bath:
    """Bath Hamiltonian h and photon-number loss rates kappa; hbar = 1.

    The vacuum kernel is K(t) = exp(-W t), with W = diag(kappa/2) + i h.
    System coupling strengths belong in Model.couplings, not in K.
    """

    h: np.ndarray
    kappa: np.ndarray

    def __post_init__(self):
        h = np.array(self.h, dtype=complex, copy=True)
        kappa = np.atleast_1d(np.array(self.kappa, dtype=float, copy=True))
        if h.ndim != 2 or h.shape[0] != h.shape[1] or not len(h):
            raise ValueError("h must be a nonempty square matrix")
        if kappa.shape != (len(h),) or np.any(kappa < 0):
            raise ValueError("Supply one nonnegative photon-number loss per mode")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(kappa)):
            raise ValueError("Bath parameters must be finite")
        if not np.allclose(h, h.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("The bath Hamiltonian must be Hermitian")
        h.flags.writeable = False
        kappa.flags.writeable = False
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "kappa", kappa)

    @property
    def W(self):
        return np.diag(self.kappa / 2) + 1j * self.h

    def kernel(self, time):
        """K_ij(t) = <a_i(t) a_j^dagger(0)> for nonnegative time."""
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be finite and nonnegative")
        return expm(-self.W * time)


def stationary_noise(bath, dt, steps, samples, seed=0):
    """Return z[time, mode, trajectory], with E[z(t)z(s)^dagger] = K(t-s).

    Exact grid samples, including nonuniform losses and complex Hermitian h.
    Noises are proper complex Gaussians. For timestep comparisons, generate
    a single fine path and subsample it; equal seeds on different grids do
    not produce nested paths.
    """
    if not np.isfinite(dt) or dt <= 0 or int(steps) != steps or steps < 0:
        raise ValueError("Invalid noise time grid")
    if int(samples) != samples or samples < 1:
        raise ValueError("samples must be a positive integer")
    rng = np.random.default_rng(seed)
    modes = len(bath.h)

    def draw():
        return (
            rng.normal(size=(modes, samples)) + 1j * rng.normal(size=(modes, samples))
        ) / np.sqrt(2)

    propagator = bath.kernel(dt)
    factor = positive_factor(np.eye(modes) - propagator @ propagator.conj().T)
    z = np.empty((steps + 1, modes, samples), complex)
    z[0] = draw()
    for step in range(steps):
        z[step + 1] = propagator @ z[step] + factor @ draw()
    return z
