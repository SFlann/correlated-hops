"""Replot the scientific panels from their archived numerical artist inputs."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


def plot(figure, output):
    folder = Path(__file__).resolve().parent / "figures" / figure
    meta = json.loads((folder / "metadata.json").read_text())
    data = np.load(folder / "arrays.npz", allow_pickle=False)
    n = len(meta["panels"])
    cols = 3 if n > 6 else min(2, n)
    rows = (n + cols - 1) // cols
    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "mathtext.fontset": "cm", "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 2.7 * rows),
        squeeze=False,
        layout="constrained",
    )
    markers = {
        "#087f8c": "o",
        "#7752a4": "s",
        "#ce6c30": "^",
        "#84929a": "D",
        "#83919b": "D",
        "#287cac": "v",
    }
    for ax, panel in zip(axes.flat, meta["panels"]):
        for band in panel["bands"]:
            a = data[band["array"]]
            ax.fill(a[:, 0], a[:, 1], color=band["colour"], alpha=band["alpha"], lw=0)
        for line in panel["lines"]:
            a = data[line["array"]]
            kw = dict(color=line["colour"], ls=line["linestyle"], lw=1.1)
            if line["coordinates"] == "mixed_axes_data":
                if np.ptp(a[:, 0]) == 0:
                    ax.axvline(a[0, 0], **kw)
                elif np.ptp(a[:, 1]) == 0:
                    ax.axhline(a[0, 1], **kw)
                continue
            if line["marker"] not in ("None", "", None):
                kw.update(marker=line["marker"], ms=2.5)
            elif line["linestyle"] == "-" and line["colour"] in markers:
                kw.update(
                    marker=markers[line["colour"]],
                    mfc="white",
                    ms=3,
                    markevery=max(1, len(a) // 5),
                )
            ax.plot(a[:, 0], a[:, 1], **kw)
        for error in panel["error_segments"]:
            ax.add_collection(
                LineCollection(
                    [data[error["array"]]], colors=error["colour"], linewidths=0.7
                )
            )
        ax.set(
            xlabel=panel["xlabel"],
            ylabel=panel["ylabel"],
            title=panel["title"],
            xlim=panel["xlim"],
            ylim=panel["ylim"],
            xscale=panel["xscale"],
            yscale=panel["yscale"],
        )
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        legend = panel["legend"] or meta["legend"]
        if legend:
            handles = [
                Line2D(
                    [],
                    [],
                    color=e["colour"],
                    ls=e["linestyle"],
                    label=e["label"],
                    marker=markers.get(e["colour"], None)
                    if e["linestyle"] == "-"
                    else None,
                    ms=3,
                    mfc="white",
                )
                for e in legend
            ]
            ax.legend(
                handles=handles,
                fontsize=6.5,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=2,
            )
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("figure")
    p.add_argument("--output", default="examples/output/paper_figure.png")
    args = p.parse_args()
    plot(args.figure, args.output)
