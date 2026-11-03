# Manuscript plotting data

This archive contains the numerical inputs to all **six main figures and
fourteen supplementary figures** in the manuscript. These are the final plotted
ensembles and reference calculations, including the 40-trajectory transverse
and exchange ensembles in main Figure 6. No simulations run when these data are
loaded or replotted.

Each `figures/Fig*/` directory contains:

- `arrays.npz`: all line coordinates, uncertainty-band vertices and error-bar
  segments, in the units actually plotted. Load with `allow_pickle=False`.
- `panel*_line*.csv`: plain-text `x,y` columns for each plotted line.
- `panel*_band*.csv`: the vertices of each filled uncertainty polygon. These
  vertices follow the boundary of the band; they are not independent samples.
- `metadata.json`: observable labels, axis ranges, line identities/styles,
  legends, the manuscript caption, and the source plotting-driver checksum.

The archive contains numerical plotting inputs extracted directly from
computed arrays. It retains the estimator and uncertainty conventions of
each figure. Confidence levels differ across figures: read the caption rather
than treating every shaded interval as one standard error. Master-equation,
HOPS and explicit-cavity stochastic reference curves remain separate.

```sh
python data/verify.py
python data/plot.py Fig3 --output examples/output/Fig3.png
```

```python
import json
import numpy as np
from pathlib import Path

folder = Path("data/figures/Fig3")
metadata = json.loads((folder / "metadata.json").read_text())
arrays = np.load(folder / "arrays.npz", allow_pickle=False)
panel = metadata["panels"][0]
curve = arrays[panel["lines"][0]["array"]]
time, correlation = curve.T
```

The convenience plotter recreates the scientific panels. It does not reproduce
the publication's artwork or final page composition. Stylistic layout revisions
do not alter the archived numerical arrays. Horizontal or vertical guide lines
with mixed data/axes coordinates are explicitly marked in the metadata.

| Figure | Content |
| --- | --- |
| 1 | Local Rabi pairs: spin entanglement and photon correlations |
| 2 | Three-spin monitoring and feedback |
| 3 | Four- and five-spin endpoint correlations |
| 4 | Five-spin controller comparisons |
| 5 | Switching feedback on and off |
| 6 | Twenty-spin transverse XXZ and exchange dynamics |
| S1–S2 | Spin readout and conditioned-hierarchy checks |
| S3–S4 | Spin–cavity pathways and a single-cavity control |
| S5 | Loss profiles and detection efficiency |
| S6 | Additional five-spin controller controls |
| S7–S10 | Photon observables, terminal readout, passive relay and bath basis controls |
| S11–S13 | Additional many-body parameters and bath observables |
| S14 | Complex hopping and timestep checks |

`manifest.json` records a SHA-256 checksum for every data file. The relative
source-driver paths are provenance identifiers, not runtime dependencies.
All data needed by the convenience plotter are inside this repository;
individual trajectory checkpoints are outside the plotting archive.

Numerical data: [CC BY 4.0](LICENSE).
