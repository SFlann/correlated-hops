"""Compare local statevector HOPS with collective-mode QuTiP HEOM.

Two resonant spins start in |10>. Each couples through X to its local vacuum
cavity. Equal photon losses allow an exact symmetric/antisymmetric rotation.
"""

import numpy as np
import qutip as qt
from correlated_hops import Bath, Model
from qutip.solver.heom import BosonicBath, HEOMSolver


def collective_bath(operator, frequency, kappa):
    # BosonicBath takes separate expansions for Re C and Im C. For t >= 0,
    # C(t) = exp[-(kappa/2 + i*frequency)*t], with unit vacuum amplitude.
    rates = [kappa / 2 + 1j * frequency, kappa / 2 - 1j * frequency]
    return BosonicBath(operator, [0.5, 0.5], rates, [-0.5j, 0.5j], rates)


def compare(hopping):
    kappa = 0.4
    eye = qt.qeye(2)
    excitation = qt.basis(2, 1).proj()
    population = qt.tensor(eye, excitation)
    hamiltonian = qt.tensor(excitation, eye) + population
    left = 0.3 * qt.tensor(qt.sigmax(), eye)
    right = 0.3 * qt.tensor(eye, qt.sigmax())
    initial = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    times = np.linspace(0, 4, 41)

    normal_modes = [
        collective_bath((left + sign * right) / np.sqrt(2), 1 + sign * hopping, kappa)
        for sign in (1, -1)
    ]

    def reference(depth):
        solver = HEOMSolver(
            hamiltonian, normal_modes, depth, options={"atol": 1e-10, "rtol": 1e-8}
        )
        return solver.run(initial.proj(), times, e_ops=[population]).expect[0]

    heom = reference(6)
    bath = Bath([[1, hopping], [hopping, 1]], [kappa, kappa])
    model = Model(hamiltonian.full(), (left.full(), right.full()), bath)
    result = model.solve(
        initial.full().ravel(), times, dt=0.02, kmax=4, samples=512, seed=17
    )
    # Each normalized trajectory contributes one population estimate per time.
    values = np.sum(abs(result.roots[:, :, [1, 3]]) ** 2, axis=2)
    mean = values.mean(axis=1)
    se = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    print(
        f"J={hopping:.2f}  final HEOM={heom[-1]:.6f}  HOPS={mean[-1]:.6f}"
        f"  final SE={se[-1]:.6f}  max |difference|={max(abs(mean - heom)):.6f}"
        f"  max HEOM depth 4->6 change={max(abs(reference(4) - heom)):.2e}"
    )


if __name__ == "__main__":
    for hopping in (0, 0.35):
        compare(hopping)
