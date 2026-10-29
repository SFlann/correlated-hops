"""Small finite-Fock-cutoff master equation for independent validation.

This explicit cavity embedding is a reference, not the HOPS implementation.
Increase photon_cutoff to check the reference's own truncation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def master_equation(model, initial, times, photon_cutoff=3):
    times = np.asarray(times, float)
    initial = np.asarray(initial, complex)
    if (
        len(times) < 2
        or times[0] != 0
        or not np.allclose(times, np.linspace(0, times[-1], len(times)))
    ):
        raise ValueError("Reference requires an equally spaced grid starting at zero")
    if int(photon_cutoff) != photon_cutoff or photon_cutoff < 1:
        raise ValueError("Invalid photon cutoff")
    m = len(model.couplings)
    d = len(initial)
    q = photon_cutoff + 1
    nb = q**m
    if d != len(model.hamiltonian) or not np.isclose(np.linalg.norm(initial), 1):
        raise ValueError("Invalid initial state")

    def kron_all(items):
        out = sparse.csr_matrix([[1.0]])
        for item in items:
            out = sparse.kron(out, item, format="csr")
        return out

    local = sparse.diags(np.sqrt(np.arange(1, q)), 1, shape=(q, q), dtype=complex)
    aa = [
        kron_all([local if j == i else sparse.eye(q) for j in range(m)])
        for i in range(m)
    ]
    h = sparse.kron(model.hamiltonian, sparse.eye(nb), format="csr")
    for i in range(m):
        h += sparse.kron(model.couplings[i], aa[i].getH()) + sparse.kron(
            model.couplings[i].conj().T, aa[i]
        )
        for j in range(m):
            h += model.bath.h[i, j] * sparse.kron(sparse.eye(d), aa[i].getH() @ aa[j])
    eye = sparse.eye(d * nb, format="csr")
    L = -1j * (sparse.kron(eye, h) - sparse.kron(h.T, eye))
    for i in range(m):
        c = np.sqrt(model.bath.kappa[i]) * sparse.kron(sparse.eye(d), aa[i])
        cdc = c.getH() @ c
        L += sparse.kron(c.conj(), c) - 0.5 * (
            sparse.kron(eye, cdc) + sparse.kron(cdc.T, eye)
        )
    psi = np.zeros(d * nb, complex)
    psi[::nb] = initial
    rho = np.outer(psi, psi.conj()).reshape(-1, order="F")
    states = expm_multiply(L.tocsr(), rho, start=0, stop=times[-1], num=len(times))
    output = []
    for state in states:
        joint = state.reshape(d * nb, d * nb, order="F").reshape(d, nb, d, nb)
        output.append(np.trace(joint, axis1=1, axis2=3))
    return np.asarray(output)
