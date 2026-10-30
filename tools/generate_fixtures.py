"""Check or regenerate Julia reference outputs using the installed Python solver.

The noise CSVs are prescribed inputs. This script does not modify them or any
manuscript data. By default it checks that the checked-in outputs are current.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from correlated_hops import solve, spin_chain
from correlated_hops.hierarchy import hierarchy
from scipy import sparse

FIXTURES = Path(__file__).resolve().parents[1] / "julia" / "test" / "fixtures"


def load_complex(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / f"{name}_real.csv", delimiter=",") + 1j * np.loadtxt(
        FIXTURES / f"{name}_imag.csv", delimiter=","
    )


def reference_outputs() -> dict[str, np.ndarray]:
    model = spin_chain(2, J=0.4, delta=1.3, g=0.3, kappa=[0.1, 0.4])
    base, annihilators, occupations = hierarchy(
        model.couplings, model.hamiltonian, model.bath.W, [1, 2]
    )
    shift = np.array([0.01 + 0.02j, -0.02 + 0.03j])
    noise_at_start = np.array([0.2 + 0.1j, -0.3 + 0.2j])
    beta = np.array([0.02 + 0.01j, 0.01 - 0.02j])
    generator = base.toarray()
    for i, coupling in enumerate(model.couplings):
        generator += (noise_at_start[i].conj() + shift[i]) * sparse.kron(
            sparse.eye(len(occupations)), coupling
        ).toarray()
        generator += beta[i] * annihilators[i].toarray()

    # Python solver: aux1, aux2, spin1, spin2. Julia fixture: interleaved.
    python_basis = list(product(range(2), range(3), range(2), range(2)))
    permutation = [
        python_basis.index((k1, k2, s1, s2))
        for s1, k1, s2, k2 in product(range(2), range(2), range(2), range(3))
    ]
    generator = generator[np.ix_(permutation, permutation)]
    noise = load_complex("noise")
    result = solve(
        model,
        [0, 0, 1, 0],
        [0, 0.1],
        dt=0.01,
        kmax=2,
        samples=1,
        noise=noise[:, :, None],
    )
    return {"generator": generator, "density": result.density[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="check outputs (the default)"
    )
    mode.add_argument("--write", action="store_true", help="replace reference outputs")
    args = parser.parse_args()
    for name, expected in reference_outputs().items():
        if args.write:
            np.savetxt(
                FIXTURES / f"{name}_real.csv", expected.real, delimiter=",", fmt="%.17e"
            )
            np.savetxt(
                FIXTURES / f"{name}_imag.csv", expected.imag, delimiter=",", fmt="%.17e"
            )
            print(f"Wrote {name} fixture")
        else:
            np.testing.assert_allclose(
                load_complex(name),
                expected,
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"Stale {name} fixture; review the numerical change before --write",
            )
            print(f"Verified current Python {name} against the Julia fixture")


if __name__ == "__main__":
    main()
