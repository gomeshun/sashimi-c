# Higher-order annihilation inputs

`SubhaloObservables(..., data_dir=...)` reads ordinary boost inputs from
`data_dir/boost` and prompt-cusp inputs from `data_dir/prompt_cusps/boost`.
The existing `SASHIMI_C_DATA_DIR` / user-cache fallback applies. Neither path
depends on the process working directory. `n=0` requires no boost tables.

An input set has `ma.txt` (M200 in Msun, one mass row per redshift), `za.txt`,
`fsh.txt`, `Bsh_0.txt` and further `Bsh_N.txt` as needed. The prompt-cusp fields
retain their historical names and survival-parameter suffixes. `manifest.json`
records schema 1, variant, calculation specification, units, full variant/core
source SHAs, the physical model contract, generation parameters and SHA-256 of
each table. Prompt-cusp contracts additionally bind the free-streaming/filter
choices, survival parameters and the two scientific spectrum file hashes.
Unattributed old tables must not be labelled as new-specification results.

The loader rejects missing files, changed hashes, incompatible model contracts,
invalid dimensions, nonfinite/negative fields, and zero fields wherever the
historical logarithmic interpolation is undefined. It preserves the original
linear `griddata` interpolation in log10 mass/redshift coordinates and log10
field values, including the existing `1e-30` addition for Bsh only. Queries
outside the covered domain fail explicitly. Successful calls expose the exact
input manifest and its hash in `last_boost_provenance`; a failed table call marks
that record invalid before reading. This field describes the last table lookup,
not a scientific calibration of the model.

The old implementation silently changed all undefined interpolations to zero.
The old ordinary generator also starts at `1e-5 Msun`, whose default resolved
subhalo interval at z=0 has zero width. That call now fails input validation.
On 2026-09-10 the user approved retaining the historical approximation only
through `annihilation_boost_factor(..., allow_incomplete_tables=True)` (and the
same keyword for the prompt-cusp method). Default calls reject incomplete tables.
The opt-in reproduces the old undefined-log-to-zero result, emits a RuntimeWarning,
and records the invalid-point count, affected-field counts, mask hash and exact
input provenance. `last_boost_provenance["valid"]` is false when approximation
was necessary. Hash, model, shape and nonfinite-table errors are never bypassed.
A new physical low-mass boundary prescription is outside the agreed scope.

The generator accepts `"allow_incomplete_tables": true` in its JSON config for
this same explicit approximation and saves the invalid counts at each order and
grid cell. For the exact ordinary z=0 zero-width mass grid, frozen A gives
fsh=Bsh=0 and luminosity ratio=1; only under the opt-in does the generator retain
this known historical zero-measure result, flagging it in the manifest. Other
invalid mass intervals still fail. Every use of an approximate table must opt in
again; the manifest does not authorize later callers automatically.

## Generating a reproducible set

The installed entry points are:

```sh
python -m boost_iteration --config boost-grid.json --data-dir /absolute/output/root
python -m boost_iteration_prompt_cusps --config cusp-grid.json --data-dir /absolute/data/root
```

The generator requires an explicit JSON configuration. This small example
exercises ordinary n=0 generation; its coarse resolution is not a scientific
recommendation and its host grid does not cover a subsequent hierarchy level.

```json
{
  "max_order": 0,
  "redshift_grid": [0.0, 0.25],
  "mass_grid_Msun": [[1e10, 1e11], [1e10, 1e11]],
  "calculation_parameters": {
    "dz": 0.25, "zmax": 1.0, "N_ma": 4, "N_herm": 2,
    "N_hermNa": 3, "logmamin": 6.0, "logmamax": 8.0
  }
}
```

Host masses are M200 **at the corresponding table redshift**. All unspecified
calculation parameters retain the current API defaults; resolved example
metadata and the complete requested configuration are saved. Prompt-cusp
configurations can additionally supply `f_surv` and `f_surv_stripped` (both
default to 1). Required spectra are read from the configured data root.

Generation uses a staging directory, refuses existing output, and publishes the
table directory only after every requested cell/order succeeds. It does not
reuse a stale `fsh.txt`. Failure leaves no partial final table. The prompt-cusp
numerical run remains deferred because the required spectra are unavailable,
as explicitly agreed with the user; no substitute spectrum was used.

Validation: all ten initial boundary tests failed before this change. Sixteen
tests now cover those failures, bitwise agreement with a frozen positive-table
observable result, four real n=0 table-generation cells, hashes, non-overwrite
cleanup after an invalid low-mass grid, frozen zero/out-of-domain results, and
explicit two-order generation with recorded invalid points. The synthetic positive table is a
software regression reference, not physical calibration. Full-source regression
and installed-artifact checks are recorded with the PR.
