"""Small spin-chain models; basis |down>, |up>, sites in left-to-right order."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import integer
from .bath import Bath
from .solver import Model


def spin_operators(n: int) -> dict[str, list[np.ndarray]]:
    """Dense site operators in the left-to-right ``|down>, |up>`` basis."""
    n = integer(n, "n")
    sm = np.array([[0, 1], [0, 0]], complex)
    sp = sm.conj().T
    local = dict(
        sm=sm, sp=sp, sx=(sm + sp) / 2, sy=(sp - sm) / (2j), sz=np.diag([-0.5, 0.5])
    )
    result = {}
    for name, a in local.items():
        result[name] = []
        for site in range(n):
            op = np.ones((1, 1), complex)
            for i in range(n):
                op = np.kron(op, a if i == site else np.eye(2))
            result[name].append(op)
    return result


def spin_chain(
    n: int = 2,
    *,
    omega_spin: float = 0.1,
    omega_cavity: float = 1.0,
    g: float = 0.3,
    counter_rotating: float = 0.25,
    J: float = 0.0,
    delta: float = 1.0,
    Jb: float = 0.25,
    kappa: ArrayLike = 0.2,
    coupling: str = "rabi",
) -> Model:
    """Independent Rabi pairs or an XXZ chain, locally coupled to hopping modes.

    coupling='rabi': F_i = g (S_i^- + counter_rotating S_i^+).
    coupling='sx'/'sz': F_i = g S_i^x/z. kappa is a NUMBER loss rate.
    """
    n = integer(n, "n")
    ops = spin_operators(n)
    H = sum(omega_spin * a for a in ops["sz"])
    for i in range(n - 1):
        H += J * (
            ops["sx"][i] @ ops["sx"][i + 1]
            + ops["sy"][i] @ ops["sy"][i + 1]
            + delta * ops["sz"][i] @ ops["sz"][i + 1]
        )
    if coupling == "rabi":
        F = tuple(g * (a + counter_rotating * b) for a, b in zip(ops["sm"], ops["sp"]))
    elif coupling in ("sx", "sz"):
        F = tuple(g * a for a in ops[coupling])
    else:
        raise ValueError("coupling must be rabi, sx or sz")
    h = omega_cavity * np.eye(n) + Jb * (np.eye(n, k=1) + np.eye(n, k=-1))
    losses = np.full(n, kappa) if np.isscalar(kappa) else np.asarray(kappa)
    return Model(H, F, Bath(h, losses))
