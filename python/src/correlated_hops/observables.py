"""Observables of the ensemble density, including nonlinear diagnostics."""

import numpy as np
from numpy.typing import ArrayLike


def expectation(density: ArrayLike, operator: ArrayLike) -> np.ndarray | np.generic:
    """Trace of ``density @ operator``, preserving leading density axes."""
    return np.einsum("...ij,ji->...", density, operator)


def connected(
    density: ArrayLike, first: np.ndarray, second: np.ndarray
) -> np.ndarray | np.generic:
    """<AB> - <A><B>, evaluated AFTER averaging trajectory projectors."""
    return expectation(density, first @ second) - expectation(
        density, first
    ) * expectation(density, second)


def partial_trace(
    density: ArrayLike, keep: list[int], dimensions: list[int]
) -> np.ndarray:
    """Keep specified subsystems in their original order; supports leading axes."""
    dims = list(dimensions)
    keep = list(keep)
    if not keep or keep != sorted(set(keep)) or min(keep) < 0 or max(keep) >= len(dims):
        raise ValueError("keep must be an increasing list of valid subsystem indices")
    d = int(np.prod(dims))
    density = np.asarray(density)
    if density.shape[-2:] != (d, d):
        raise ValueError("Density dimensions do not match")
    leading = density.shape[:-2]
    a = density.reshape(*leading, *dims, *dims)
    offset = len(leading)
    for site in reversed(range(len(dims))):
        if site not in keep:
            a = np.trace(a, axis1=offset + site, axis2=offset + len(dims) + site)
            dims.pop(site)
    d = int(np.prod(dims))
    return a.reshape(*leading, d, d)


def concurrence(density: ArrayLike) -> np.ndarray | np.generic:
    """Wootters concurrence of a normalised, positive two-qubit density."""
    rho = np.asarray(density, complex)
    if rho.shape[-2:] != (4, 4):
        raise ValueError("Concurrence requires a two-qubit density")
    if not np.allclose(rho, rho.conj().swapaxes(-1, -2), atol=1e-9):
        raise ValueError("Density is not Hermitian")
    if not np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1, atol=1e-8):
        raise ValueError("Density must have unit trace")
    eig, u = np.linalg.eigh(rho)
    if eig.min() < -1e-8:
        raise ValueError("Density is not positive")
    root = (u * np.sqrt(np.maximum(eig, 0))[..., None, :]) @ u.conj().swapaxes(-1, -2)
    sy = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sy, sy)
    r = root @ yy @ rho.conj() @ yy @ root
    values = np.sqrt(
        np.maximum(np.linalg.eigvalsh((r + r.conj().swapaxes(-1, -2)) / 2), 0)
    )
    return np.maximum(0, values[..., -1] - values[..., :-1].sum(-1))
