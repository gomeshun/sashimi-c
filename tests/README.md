# Tests

Run from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

CI runs the same tests on Python 3.10–3.13. Tests cover independent equation and
numerical references, public solver behavior, boundary conditions and relevant
catalog or saved-output regressions. Passing them does not establish simulation
calibration or observational constraints.

The solver tests compare against an analytic host model and direct ODE
integration. They require no stored catalog arrays. The optional
`benchmarks/benchmark_picard_tidal_stripping.py` measures an explicitly configured
table; it is not a timing of every default catalog setting.

## Repository contents

Keep maintained tests, small required fixtures and user-facing numerical notes
in the repository. Put local experiments, generated catalogs, timing logs and
temporary validation reports under the ignored `/_scratch/` directory. Retire
one-off experiment drivers after their useful checks become maintained tests.
The old generated `/validation/maintenance/` location is also ignored.

The [development evidence](https://github.com/gomeshun/sashimi-c/tree/c8f80295e86f61dcf39ee3d427217d3ae12099ba/validation/maintenance) remains available
at the fixed pre-cleanup commit.
