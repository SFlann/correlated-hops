"""Long-time correlations and entanglement through communicating vacuum baths."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from correlated_hops import (
    __version__,
    concurrence,
    connected,
    spin_chain,
    spin_operators,
)
from correlated_hops.reference import master_equation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PARAMETERS = dict(
    n=2,
    omega_spin=0.1,
    omega_cavity=1.0,
    g=0.3,
    counter_rotating=0.25,
    J=0.0,
    delta=1.0,
    kappa=[0.1, 0.4],
    coupling="rabi",
)


def connected_sem(roots, first, second):
    """Delta-method SEM with covariance between the three raw moments."""
    a = np.einsum("tsi,ij,tsj->ts", roots.conj(), first, roots).real
    b = np.einsum("tsi,ij,tsj->ts", roots.conj(), second, roots).real
    ab = np.einsum("tsi,ij,tsj->ts", roots.conj(), first @ second, roots).real
    influence = ab - b.mean(1)[:, None] * a - a.mean(1)[:, None] * b
    return influence.std(axis=1, ddof=1) / np.sqrt(roots.shape[1])


def observables(density):
    sz = spin_operators(2)["sz"]
    return np.stack([connected(density, *sz).real, concurrence(density)], axis=-1)


def bootstrap_interval(batch_densities, repetitions=1000):
    """Resample independent equal-size batches, then evaluate nonlinear observables."""
    count = len(batch_densities)
    rng = np.random.default_rng(81231)
    weights = rng.multinomial(count, np.full(count, 1 / count), size=repetitions)
    densities = np.einsum("rb,btij->rtij", weights / count, batch_densities)
    return np.quantile(observables(densities), [0.025, 0.975], axis=0)


def qualify(model, initial, times, reference):
    """Independent reference cutoff and nested-noise dt/hierarchy checks."""
    higher_reference = master_equation(model, initial, times, photon_cutoff=6)
    noise = model.bath.sample(0.0025, round(times[-1] / 0.0025), 64, seed=91023)
    fine = model.solve(initial, times, dt=0.005, kmax=4, samples=64, noise=noise)
    coarse = model.solve(initial, times, dt=0.01, kmax=4, samples=64, noise=noise[::2])
    deeper = model.solve(initial, times, dt=0.005, kmax=6, samples=64, noise=noise)
    comparisons = {
        "reference_cutoff_4_to_6": (reference, higher_reference),
        "nested_dt_0.01_to_0.005": (coarse.density, fine.density),
        "hierarchy_4_to_6": (fine.density, deeper.density),
    }
    checks = {}
    for name, (a, b) in comparisons.items():
        delta = np.max(abs(observables(a) - observables(b)), axis=0)
        checks[name] = dict(
            max_density_entry_change=float(np.max(abs(a - b))),
            max_Gzz_change=float(delta[0]),
            max_concurrence_change=float(delta[1]),
        )
        # Observable-specific numerical gates, independent of sampling agreement.
        if np.max(abs(a - b)) > 3e-4 or delta[0] > 3e-4 or delta[1] > 2e-3:
            raise RuntimeError(
                f"Numerical qualification failed: {name}: {checks[name]}"
            )
    return checks


def run(out, samples=4096, batch_size=256, t_end=24.0, validate=False):
    """Preserve the original model; extend duration and a fixed trajectory count."""
    if batch_size < 1 or samples < 4 * batch_size or samples % batch_size:
        raise ValueError(
            "Use at least four equal batches; samples must divide by batch-size"
        )
    if t_end <= 0 or not np.isclose(t_end / 0.2, round(t_end / 0.2)):
        raise ValueError("t-end must be a positive multiple of 0.2")
    out.mkdir(parents=True, exist_ok=True)
    initial = np.array([1, 0, 0, 0], complex)
    times = np.linspace(0, t_end, round(t_end / 0.2) + 1)
    records = {
        "versions": dict(
            correlated_hops=__version__,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
        ),
        "uncertainty": "Pointwise 95% percentile intervals; 1000 bootstrap resamples of independent equal-size trajectory batches, observables computed AFTER density averaging",
        "bootstrap_seed": 81231,
        "samples_per_case": samples,
        "batch_size": batch_size,
        "reference_photon_cutoff": 4,
        "preparation": "both spins down; factorised vacuum bath",
        "physical_model": "Unchanged from original short-time README example",
        "qualification_requested": validate,
        "runs": [],
    }
    arrays = {"times": times}
    for index, (hopping, colour) in enumerate([(0.0, "#087f8c"), (0.25, "#7752a4")]):
        parameters = PARAMETERS | {"Jb": hopping}
        model = spin_chain(**parameters)
        reference = master_equation(model, initial, times, photon_cutoff=4)
        checks = qualify(model, initial, times, reference) if validate else None
        batches, diagnostics = [], []
        for batch in range(samples // batch_size):
            result = model.solve(
                initial,
                times,
                dt=0.01,
                kmax=4,
                samples=batch_size,
                seed=17 + 1000 * batch,
            )
            batches.append(result.density)
            diagnostics.append(result.diagnostics)
            print(
                f"Jb={hopping:g}: {(batch + 1) * batch_size}/{samples} trajectories",
                flush=True,
            )
        batches = np.asarray(batches)
        density = batches.mean(axis=0)
        value, exact = observables(density), observables(reference)
        interval = bootstrap_interval(batches)
        records["runs"].append(
            dict(
                parameters=parameters,
                batches=diagnostics,
                qualification=checks,
                maximum_abs_error=dict(
                    zip(
                        ["Gzz", "concurrence"],
                        np.max(abs(value - exact), axis=0).tolist(),
                    )
                ),
                maximum_reference=dict(
                    zip(["Gzz", "concurrence"], exact.max(axis=0).tolist())
                ),
            )
        )
        arrays.update(
            {
                f"case{index}_density": density,
                f"case{index}_batch_densities": batches,
                f"case{index}_reference": reference,
                f"case{index}_interval": interval,
            }
        )
    render(arrays, out)
    np.savez_compressed(out / "rabi_pair.npz", **arrays)
    (out / "rabi_pair.json").write_text(json.dumps(records, indent=2) + "\n")
    print(out / "rabi_pair.png", flush=True)


def render(arrays, out):
    """Replot saved complex densities and uncertainty without rerunning dynamics."""
    out.mkdir(parents=True, exist_ok=True)
    times = arrays["times"]
    t_end = times[-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for index, (hopping, colour) in enumerate([(0.0, "#087f8c"), (0.25, "#7752a4")]):
        value = observables(arrays[f"case{index}_density"])
        exact = observables(arrays[f"case{index}_reference"])
        interval = arrays[f"case{index}_interval"]
        for col, ax in enumerate(axes):
            ax.fill_between(
                times,
                interval[0, :, col],
                interval[1, :, col],
                color=colour,
                alpha=0.2,
                lw=0,
            )
            ax.plot(
                times, value[:, col], color=colour, lw=1.8, label=rf"$J_b={hopping:g}$"
            )
            ax.plot(times, exact[:, col], ":", color=colour, lw=2)
    for letter, ax, title, ylabel in zip(
        "ab",
        axes,
        [
            "Bath-mediated correlations",
            "Two-qubit entanglement",
        ],
        [r"Connected correlation $G_{zz}^{1,2}$", "Two-qubit concurrence"],
    ):
        ax.set(xlabel=r"Time $\omega_c t$", ylabel=ylabel, xlim=(0, t_end))
        ax.set_title(f"({letter}) {title}", fontsize=10, loc="left", pad=10)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    handles = [
        Line2D([], [], color=c, lw=2, label=rf"$J_b={j:g}$")
        for j, c in [(0, "#087f8c"), (0.25, "#7752a4")]
    ]
    handles += [
        Line2D([], [], color=".25", label="HOPS"),
        Line2D([], [], color=".25", ls=":", label="Master equation"),
        Patch(facecolor=".5", alpha=0.2, label="95% interval"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=5, loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2)
    fig.savefig(out / "rabi_pair.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--t-end", type=float, default=24.0)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check reference cutoff, hierarchy and nested-noise timestep",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        help="Replot an existing NPZ without running dynamics",
    )
    args = parser.parse_args()
    if args.input_data is not None:
        with np.load(args.input_data) as arrays:
            render(arrays, args.output_dir)
    else:
        run(args.output_dir, args.samples, args.batch_size, args.t_end, args.validate)


if __name__ == "__main__":
    main()
