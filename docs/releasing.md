# Releasing the companion package

Run these checks from a clean checkout of the intended release commit.
Install the optional tooling with `python -m pip install -e '.[plot,test,dev]'`.

```sh
ruff check .
ruff format --check .
pytest
python tools/generate_fixtures.py --check
python data/verify.py
python tools/verify_example_assets.py
python examples/python/rabi_pair.py --validate
python examples/python/noise_convergence.py
python data/plot.py Fig3 --output examples/output/Fig3.png
python -m build
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=julia examples/julia/spin_chain.jl
```

CI builds a wheel from the source distribution, installs that wheel and runs
the Python tests and fixture check against it. It also executes both examples
and a representative archive replot. Compatibility checks include Python 3.10
and 3.12, Julia 1.10 and 1.12, and a separate Python dependency-floor job.

The Python source distribution includes tests, examples, documentation,
fixture tooling and the small Julia implementation. The wheel contains the
Python library. The manuscript plotting archive is distributed through the
GitHub source checkout/archive, with its own licence and checksums.

Before publishing:

1. Confirm CI succeeds for the intended commit and inspect the example image.
2. Confirm the manuscript version matches the figure archive and its captions.
   Do not regenerate manuscript inputs from companion examples.
3. Update the version consistently in `pyproject.toml`,
   `python/src/correlated_hops/__init__.py`, `julia/Project.toml`,
   `CITATION.cff` and `CHANGELOG.md`. Record the actual release date.
4. Create the version tag and GitHub release from that tested commit. Include
   the scope, installation instructions and links to the paper and data guide.
5. For the paper's persistent citation, archive the release with the chosen
   research archive and add its identifier to the citation metadata.

Visibility changes, release publication and uploading to a package registry
are separate publication actions. The build/check commands above perform none
of them.
