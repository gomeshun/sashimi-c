# Package layout and physical examples

The migration candidate uses a `src` layout. Install the project before importing it;
Python is not expected to import the checkout just because it is the working directory.

| Location | Purpose |
| --- | --- |
| `src/sashimi_c/` | Standard public API and model-specific implementation |
| `src/sashimi_c/_physics.py` | Existing physical prescriptions and observable kernels |
| `src/sashimi_c/_itamae_components.py` | Model stages used by ITAMAE population execution |
| `src/*.py` | Compatibility import modules forwarding to the package; no duplicate model implementation |
| `notebooks/usage_walkthrough.ipynb` | Executed physical example, including subhalo mass functions and weighted Vmax-rmax distributions |
| `notebooks/scientific_validation.ipynb` | Independent migration and numerical-validation evidence |
| `notebooks/archive/` | Preserved historical examples and intermediate migration demonstrations |
| `tests/` | Automated regression tests and their frozen references |
| `docs/` | API, physics and validation explanations |
| `scripts/` | Development, notebook execution and family-validation commands |
| `validation/` | Recorded scientific evidence, separate from importable runtime code |

The standard `import sashimi_c` and previously distributed helper imports remain
available. Compatibility modules refer to the same module objects as their package
counterparts, so there is one implementation and one module state. Numerical
prescriptions, units, thresholds and solver defaults are unchanged by this move.

## Run the physical example

From this checkout, `uv sync --extra demo` resolves the exact ITAMAE source in
`pyproject.toml` and `uv.lock`. Register/select that environment's Jupyter kernel,
then run the walkthrough from the first cell. Its introduction gives the commands.
For coordinated local checkouts, use `uv pip install --no-sources -e ../itamae -e '.[demo]'`.
The distribution providing `import itamae` is `sashimi-itamae`.

The example uses a 1e12-Msun host, explicit accretion-mass/redshift bounds, a
current-bound-mass selection for structure, and expected-count weights. Figure
axes specify km/s, kpc and solar masses. A joint grid refinement measures the
sensitivity of the displayed calculation; it is not a certificate of simulation
calibration or an observation-derived constraint. Notebook output files go to
`notebooks/outputs/` and are ignored by Git. Executed figures remain in the notebook.

Source distributions include the examples and validation bundle. Wheels contain
the importable package, required runtime data and compatibility modules. Verify
both installed wheels and wheels rebuilt from source distributions outside the
checkout; a source-tree import alone is not a packaging test.
