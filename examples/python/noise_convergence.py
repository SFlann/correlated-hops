"""Watch the public noise sampler recover a ten-site bath correlation kernel.

All checkpoints are nested prefixes of one fixed ensemble. No exact values
enter the stochastic estimator. Optional GIF rendering needs Pillow (installed
with the plot extra); no external video encoder is required.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy
from correlated_hops import Bath, __version__
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

SITES = np.array([0, 1, 3, 7])
COLOURS = ["#087f8c", "#d77a2a", "#7752a4", "#28455b"]
SEED = 20260921


def sample(out, samples=100_000, batch_size=512):
    """Stream bounded batches through Bath.sample; retain complex estimates."""
    started = perf_counter()
    times = np.arange(201) * 0.1
    h = 3 * np.eye(10) + 0.5 * (np.eye(10, k=1) + np.eye(10, k=-1))
    bath = Bath(h, np.full(10, 0.1))
    counts = np.unique(
        np.r_[
            np.rint(np.geomspace(1, samples, 50)).astype(int),
            [n for n in [10, 100, 1000, 10000] if n <= samples],
        ]
    )
    sums = np.zeros((len(times), len(SITES)), complex)
    estimates = []
    check_steps = [0, 20, 50, 100, 200]
    dimension = len(check_steps) * len(SITES)
    covariance_sum = np.zeros((dimension, dimension), complex)
    pseudo_sum = np.zeros_like(covariance_sum)
    mean_sum = np.zeros(dimension, complex)
    done, checkpoint = 0, 0
    seeds = np.random.SeedSequence(SEED).spawn((samples + batch_size - 1) // batch_size)
    example = None
    for child_seed in seeds:
        size = min(batch_size, samples - done)
        # Each batch owns a fixed independent SeedSequence child.
        seed = int(child_seed.generate_state(1)[0])
        z = bath.sample(0.1, 200, size, seed=seed)
        if example is None:
            example = z[:, SITES, 0].copy()
        products = z[:, SITES, :] * z[0, 0, :].conj()
        while checkpoint < len(counts) and counts[checkpoint] <= done + size:
            prefix = counts[checkpoint] - done
            estimates.append(
                (sums + products[:, :, :prefix].sum(axis=-1)) / counts[checkpoint]
            )
            checkpoint += 1
        sums += products.sum(axis=-1)
        joint = z[check_steps][:, SITES].reshape(dimension, size)
        covariance_sum += joint @ joint.conj().T
        pseudo_sum += joint @ joint.T
        mean_sum += joint.sum(axis=1)
        done += size
    exact = np.array([bath.kernel(t)[SITES, 0] for t in times])
    estimates = np.asarray(estimates)
    errors = np.sqrt(np.mean(abs(estimates - exact) ** 2, axis=(1, 2)))
    selected_times = times[check_steps]
    target = np.block(
        [
            [
                (
                    (bath.kernel(t - s) if t >= s else bath.kernel(s - t).conj().T)[
                        np.ix_(SITES, SITES)
                    ]
                )
                for s in selected_times
            ]
            for t in selected_times
        ]
    )
    checks = dict(
        complex_rms_error=float(errors[-1]),
        complex_max_error=float(np.max(abs(estimates[-1] - exact))),
        joint_covariance_max_error=float(
            np.max(abs(covariance_sum / samples - target))
        ),
        pseudocovariance_max=float(np.max(abs(pseudo_sum / samples))),
        mean_max=float(np.max(abs(mean_sum / samples))),
    )
    if max(checks.values()) > 8 / np.sqrt(samples):
        raise RuntimeError(f"Fixed noise validation tolerance failed: {checks}")
    data = dict(
        times=times,
        counts=counts,
        estimates=estimates,
        exact=exact,
        errors=errors,
        example=example,
        sites=SITES + 1,
        joint_covariance=covariance_sum / samples,
        joint_target=target,
        pseudocovariance=pseudo_sum / samples,
    )
    np.savez_compressed(out / "noise_convergence.npz", **data)
    record = dict(
        model=dict(modes=10, omega_c=3.0, Jb=0.5, kappa=0.1, dt=0.1, steps=200),
        samples=samples,
        batch_size=batch_size,
        seed=SEED,
        seed_scheme="SeedSequence children; one independent integer seed per batch",
        estimator="mean(z_j(t) * conj(z_1(0))); magnitude AFTER complex averaging",
        units="Absolute error; unit equal-time variance; not relative error along curves",
        checks=checks,
        elapsed_seconds=perf_counter() - started,
        versions=dict(
            correlated_hops=__version__,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
        ),
        checkpoints=[
            dict(samples=int(n), complex_rms_error=float(e))
            for n, e in zip(counts, errors)
        ],
    )
    (out / "noise_convergence.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(checks), flush=True)
    return data


def decorate(ax):
    ax.set(xlim=(0, 20), xlabel=r"Time lag $\tau$", ylabel=r"$|\widehat K_{j1}(\tau)|$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)


def snapshots(data, out):
    targets = [10, 100, 1000, int(data["counts"][-1])]
    indices = [int(np.argmin(abs(data["counts"] - n))) for n in targets]
    ymax = max(1.05, float(abs(data["estimates"][indices]).max()) * 1.05)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, sharey=True)
    for letter, ax, index in zip("abcd", axes.flat, indices):
        for j, colour in enumerate(COLOURS):
            ax.plot(
                data["times"], abs(data["estimates"][index, :, j]), color=colour, lw=1.4
            )
            ax.plot(data["times"], abs(data["exact"][:, j]), color=colour, ls=":", lw=2)
        decorate(ax)
        ax.set_ylim(-0.025, ymax)
        ax.set_title(
            f"({letter}) {data['counts'][index]:,} trajectories",
            loc="left",
            fontsize=11,
        )
    handles = [
        Line2D([], [], color=c, label=f"Site {j}") for j, c in zip(SITES + 1, COLOURS)
    ]
    handles += [
        Line2D([], [], color=".25", label="Sampled"),
        Line2D([], [], color=".25", ls=":", label="Exact kernel"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(out / "noise_convergence_panels.png", dpi=160)
    plt.close(fig)


def animate(data, out):
    indices = np.flatnonzero(data["counts"] >= 10)
    ymax = max(1.05, float(abs(data["estimates"][indices]).max()) * 1.05)
    fig, (ax, error_ax) = plt.subplots(
        1, 2, figsize=(9.0, 3.9), gridspec_kw=dict(width_ratios=[2.1, 1])
    )
    lines = []
    for j, colour in enumerate(COLOURS):
        ax.plot(data["times"], abs(data["exact"][:, j]), color=colour, ls=":", lw=2)
        (line,) = ax.plot([], [], color=colour, lw=1.5, label=f"Site {SITES[j] + 1}")
        lines.append(line)
    decorate(ax)
    ax.set_ylim(-0.025, ymax)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    error_ax.loglog(
        data["counts"],
        1 / np.sqrt(data["counts"]),
        "--",
        color=".6",
        label=r"$N^{-1/2}$ scale",
    )
    (error_line,) = error_ax.loglog(
        [], [], color="#28455b", lw=2, label="Complex RMS error"
    )
    error_ax.set(
        xlim=(10, int(data["counts"][-1])),
        ylim=(0.001, 1),
        xlabel="Trajectories N",
        ylabel="Absolute error",
    )
    error_ax.spines[["top", "right"]].set_visible(False)
    error_ax.grid(alpha=0.18)
    error_ax.legend(frameon=False, fontsize=8, loc="upper right")
    title = fig.suptitle("", fontsize=13)
    fig.text(
        0.5,
        0.025,
        "Solid: sampled correlations    Dotted: exact kernel    Same cumulative ensemble",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    def update(index):
        for j, line in enumerate(lines):
            line.set_data(data["times"], abs(data["estimates"][index, :, j]))
        error_line.set_data(data["counts"][: index + 1], data["errors"][: index + 1])
        title.set_text(
            f"Random noise recovers the bath kernel  |  N = {data['counts'][index]:,}"
        )
        return [*lines, error_line, title]

    frames = [*indices, *([indices[-1]] * 12)]
    movie = FuncAnimation(fig, update, frames=frames, interval=250, blit=False)
    movie.save(out / "noise_convergence.gif", writer=PillowWriter(fps=4), dpi=105)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument(
        "--input-data",
        type=Path,
        help="Render existing arrays to output-dir without sampling",
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    if args.samples < 2000 or args.batch_size < 1:
        parser.error("Use samples >= 2000 and a positive batch-size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_data or args.input_data is not None:
        with np.load(
            args.input_data or args.output_dir / "noise_convergence.npz"
        ) as saved:
            data = dict(saved)
    else:
        data = sample(args.output_dir, args.samples, args.batch_size)
    snapshots(data, args.output_dir)
    if not args.no_animation:
        animate(data, args.output_dir)
    print(args.output_dir / "noise_convergence_panels.png", flush=True)


if __name__ == "__main__":
    main()
