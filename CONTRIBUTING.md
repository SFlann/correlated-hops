# Contributing

Please open an issue describing the physical model and expected behaviour
before proposing a substantial new solver feature. Include a small example,
software versions, seed, timestep, hierarchy bounds and, for MPS calculations,
cutoff and bond cap when reporting numerical problems.

Changes to dynamics should include a small independent reference calculation.
Use nested noise paths for time-step tests, and preserve failed trajectories
when diagnosing a stochastic calculation. Do not silently discard samples or
substitute a different loss profile for an existing comparison.

Run `pytest` and `julia --project=julia -e 'using Pkg; Pkg.test()'` before a pull
request. Keep plotting-data changes separate from solver changes and update
the data manifest whenever a numerical source changes.
