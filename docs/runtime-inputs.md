# Runtime inputs and reproducible Monte Carlo catalogs

`SubhaloObservables.subhalo_catalog_MC(mth, seed=...)` and `rng=Generator(...)`
use ITAMAE's independent Poisson multiplicities. They do not read or modify
NumPy's global random state. The two forms are mutually exclusive. With neither,
a fresh local generator makes a nondeterministic draw. Rows are grouped in input
node order. The point-process distribution equals the former Poisson total plus
categorical allocation; identical integer seeds are not promised to reproduce
old implementation row sequences. The eight tuple columns retain their existing
units and meanings. Empty selections produce empty arrays without division by
zero.

Prompt-cusp models accept `data_dir`. It takes precedence over
`SASHIMI_C_DATA_DIR`; the default is `$XDG_CACHE_HOME/sashimi-c`, or
`~/.cache/sashimi-c`. Input locations do not depend on the process working
directory. Required tables are checked and a `RuntimeError` names the missing
file. A spectrum is never substituted or downloaded automatically.

The existing prompt-cusp calculations require both the CAMB input
`Planck2018_CAMB_extrap.dat` and the redshift-31 `powerspectrum31.txt` table.
They are absent from this repository and the checked upstream tree. The user
explicitly authorized deferring calculations requiring the unavailable table on
2026-09-10 and requested an explicit runtime error. Physical prompt-cusp
validation is therefore deferred, not counted as passed. The prompt-cusp Monte
Carlo cache also uses the configured data root unless given an explicit path.
Higher-order boost-table preparation is a separate runtime-data deliverable.

All 53 tests pass, including three new regressions that failed before the
change: deterministic/local random state, empty MC and invalid arguments, and
missing spectra from a working directory outside the source tree. Existing
catalog and observable numerical regressions are unchanged.
