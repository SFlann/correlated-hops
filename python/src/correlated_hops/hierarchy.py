"""Sparse factorially scaled hierarchy with individual local bounds."""

from itertools import product

import numpy as np
from scipy import sparse


def hierarchy(couplings, hamiltonian, W, kmax):
    occupations = np.array(list(product(*(range(k + 1) for k in kmax))), dtype=int)
    lookup = {tuple(k): i for i, k in enumerate(occupations)}
    nh = len(occupations)
    d = hamiltonian.shape[0]
    annihilators = []
    for j in range(len(kmax)):
        rows, cols, values = [], [], []
        for col, k in enumerate(occupations):
            if k[j]:
                lower = k.copy()
                lower[j] -= 1
                rows.append(lookup[tuple(lower)])
                cols.append(col)
                values.append(np.sqrt(k[j]))
        annihilators.append(sparse.csr_matrix((values, (rows, cols)), shape=(nh, nh)))
    eye = sparse.eye(d, format="csr", dtype=complex)
    ih = sparse.eye(nh, format="csr", dtype=complex)
    base = sparse.kron(ih, -1j * hamiltonian, format="csr")
    for i, b in enumerate(annihilators):
        base += sparse.kron(b.getH(), couplings[i], format="csr")
        base -= sparse.kron(b, couplings[i].conj().T, format="csr")
        for j, bj in enumerate(annihilators):
            if W[i, j] != 0:
                base -= W[i, j] * sparse.kron(b.getH() @ bj, eye, format="csr")
    return (
        base.tocsr(),
        [sparse.kron(b, eye, format="csr") for b in annihilators],
        occupations,
    )
