# Contributing

Please open an issue describing the physical model and expected behaviour
before proposing a substantial new solver feature. Include a small example,
software versions, seed, timestep, hierarchy bounds and, for MPS calculations,
cutoff and bond cap when reporting numerical problems.

Changes to dynamics should include a small independent reference calculation.
Use nested noise paths for time-step tests, and preserve failed trajectories
when diagnosing a stochastic calculation. Do not silently discard samples or
substitute a different loss profile for an existing comparison.

Install the development tools with `python -m pip install -e '.[plot,test,dev]'`.
Run `ruff check .`, `ruff format --check .`, `pytest`,
`python tools/generate_fixtures.py --check` and
`julia --project=julia -e 'using Pkg; Pkg.test()'` before a pull request.
Format Julia sources with the separate, pinned tooling environment and the
repository's `.JuliaFormatter.toml` configuration:

```sh
julia --project=tools/julia -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; format(["julia/src", "julia/test", "examples/julia"])'
```

Use `format([...]; overwrite=false)` to check formatting without changing files.
The formatter is a development dependency and is not loaded by the solver.

Changes to input validation or numerical diagnostics should include a regression
test of the public behaviour. Preserve the public array/basis conventions in
[the API guide](docs/api.md). The cross-language fixture command checks the
current Python implementation without modifying its prescribed noise inputs;
see [fixture provenance](julia/test/fixtures/README.md) before regenerating outputs.

Keep plotting-data changes separate from solver changes and update the data
manifest whenever a numerical source changes. See [release checks](docs/releasing.md)
for distribution builds, examples and the publication checklist.
