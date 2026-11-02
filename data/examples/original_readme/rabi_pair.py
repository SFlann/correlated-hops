"""Compare communicating Rabi pairs with an independent finite-cutoff reference."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from correlated_hops import __version__, connected, solve, spin_chain, spin_operators
from correlated_hops.reference import master_equation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def connected_sem(
    roots: np.ndarray, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Pointwise delta-method SEM, retaining covariance between the raw moments."""
    a = np.einsum("tsi,ij,tsj->ts", roots.conj(), first, roots).real
    b = np.einsum("tsi,ij,tsj->ts", roots.conj(), second, roots).real
    ab = np.einsum("tsi,ij,tsj->ts", roots.conj(), first @ second, roots).real
    influence = ab - b.mean(1)[:, None] * a - a.mean(1)[:, None] * b
    return influence.std(axis=1, ddof=1) / np.sqrt(roots.shape[1])


def run(out: Path) -> None:
    """Write a demo plot and its parameters, versions and numerical diagnostics."""
    out.mkdir(parents=True, exist_ok=True)
    initial = np.array([1, 0, 0, 0], complex)
    times = np.linspace(0, 2, 41)
    fig, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
    records = {
        "versions": {
            "correlated_hops": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "uncertainty": "Pointwise delta-method +/-1 SEM across 256 independent trajectories",
        "reference_photon_cutoff": 4,
        "preparation": "both spins down; factorised vacuum bath",
        "runs": [],
    }
    for hopping, colour, marker in [(0, "#087f8c", "o"), (0.25, "#7752a4", "s")]:
        parameters = dict(
            n=2,
            omega_spin=0.1,
            omega_cavity=1.0,
            g=0.3,
            counter_rotating=0.25,
            J=0.0,
            delta=1.0,
            Jb=hopping,
            kappa=[0.1, 0.4],
            coupling="rabi",
        )
        model = spin_chain(**parameters)
        result = solve(model, initial, times, dt=0.01, kmax=4, samples=256, seed=17)
        reference = master_equation(model, initial, times, photon_cutoff=4)
        sz = spin_operators(2)["sz"]
        value = connected(result.density, *sz).real
        sem = connected_sem(result.roots, *sz)
        exact = connected(reference, *sz).real
        ax.fill_between(times, value - sem, value + sem, color=colour, alpha=0.2, lw=0)
        ax.plot(
            times,
            value,
            color=colour,
            marker=marker,
            markevery=8,
            ms=3,
            mfc="white",
            label=f"$J_b = {hopping:g}$",
        )
        ax.plot(times, exact, ":", color=colour)
        records["runs"].append(
            {"parameters": parameters, "diagnostics": result.diagnostics}
        )
    handles, _ = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], color="0.25", label="HOPS"),
            Line2D([], [], color="0.25", ls=":", label="Master equation"),
            Patch(facecolor="0.5", alpha=0.2, label=r"Pointwise $\pm 1$ SEM"),
        ]
    )
    ax.set(
        xlabel=r"Time $\omega_c t$", ylabel=r"Connected spin correlation $G_{zz}^{1,2}$"
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out / "rabi_pair.png", dpi=180)
    plt.close(fig)
    (out / "rabi_pair.json").write_text(json.dumps(records, indent=2) + "\n")
    print(out / "rabi_pair.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
    )
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
