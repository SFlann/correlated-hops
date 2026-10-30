# Cross-language fixtures

These CSVs use the Python basis convention: the last tensor index varies fastest.
Real and imaginary parts are stored separately at double precision.

- `noise_*.csv` is a prescribed half-step noise path, with rows as times and
  columns as the two bath modes. It is an input, not an expected solver output.
- `generator_*.csv` is the complete generator for a two-spin model with
  `J=0.4`, `delta=1.3`, `g=0.3`, number losses `[0.1, 0.4]` and hierarchy
  bounds `[1, 2]`. Its basis is `spin1, aux1, spin2, aux2`. The prescribed
  shift, noise and nonlinear coefficients are explicit in the generator script.
- `density_*.csv` is the Python nonlinear root projector at time `0.1`,
  starting in `|up, down>`, with `dt=0.01`, bounds `[2, 2]`, and that noise.

From the repository root, with the Python package installed:

```sh
python tools/generate_fixtures.py --check
```

CI checks these outputs against the current Python code and then tests Julia
against the same files. After an intentional numerical change, inspect its
independent reference tests before running `--write` and reviewing the CSV diff.
The noise inputs and manuscript archive are never rewritten by this command.
