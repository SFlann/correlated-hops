"""A short, inexpensive example; increase samples and cutoffs for research."""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from correlated_hops import solve, spin_chain, spin_operators, connected
from correlated_hops.reference import master_equation

out = Path(__file__).resolve().parents[1] / "output"
out.mkdir(exist_ok=True)
initial = np.array([1, 0, 0, 0], complex)
times = np.linspace(0, 2, 41)
fig, ax = plt.subplots(figsize=(5, 3.2))
records = {}
for hopping, colour, marker in [(0, "#087f8c", "o"), (0.25, "#7752a4", "s")]:
    model = spin_chain(2, g=0.3, Jb=hopping, kappa=[0.1, 0.4])
    result = solve(model, initial, times, dt=0.01, kmax=4, samples=256, seed=17)
    reference = master_equation(model, initial, times, photon_cutoff=4)
    sz = spin_operators(2)["sz"]
    value = connected(result.density, *sz).real
    exact = connected(reference, *sz).real
    ax.plot(
        times,
        value,
        color=colour,
        marker=marker,
        markevery=8,
        ms=3,
        mfc="white",
        label=f"Jb = {hopping:g}",
    )
    ax.plot(times, exact, ":", color=colour)
    records[str(hopping)] = result.diagnostics
ax.set(xlabel="Time", ylabel="Connected spin correlation")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "rabi_pair.png", dpi=180)
(out / "rabi_pair.json").write_text(json.dumps(records, indent=2) + "\n")
print(out / "rabi_pair.png")
