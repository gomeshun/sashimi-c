# Resolution and state validation — 2026-09-11

One variable is refined at a time: M0=1e12 Msun, observation z=0, zmax=3,
log accretion M200/Msun=6–10, baseline N_ma=16, dz=.25, concentration Hermite
order 3, host Hermite order 8, ct_th=.77 and pert2_shanks. These reduced settings
are not the complete default configuration. The finest grid is a finite
comparator, not continuum truth, and no error budget is inferred for all
parameter choices. All source SHAs, original warnings/failures, settings and
full arrays remain in the family science bundle; summaries and hashes are in
`validation/science`. Displayed metrics were recomputed from every saved
catalog. Finite arrays, nonnegative weights and m_bound <= fitted Mvir_acc
were verified throughout. Repeated baseline and post-EPS catalogs match bitwise.

| Refinement | Count change [%] | Bound mass fraction change [%] | Other metric change [%] |
| --- | ---: | ---: | ---: |
| N_ma-256 → N_ma-500 | -5.36668e-07 | 0.503829 | 0.542911 |
| dz-0.01 → dz-0.005 | 0.258306 | 0.753592 | 0.337098 |
| N_herm-5 → N_herm-7 | 0 | -1.11022e-14 | 8.91913e-05 |
| N_hermNa-64 → N_hermNa-200 | 0.00453292 | 0.00375322 | 0.00281248 |
| baseline → odeint | 0 | -3.82061 | -1.99156 |
| odeint → odeint-tight | 0 | -1.49779e-06 | -1.33858e-06 |

The final column is the weighted NFW luminosity proxy (not the full higher-order boost).

The larger baseline-to-ODE difference is a stripping-approximation effect.
Tightening ODE tolerances changes these metrics by about 1e-6 percent, so it
does not explain the roughly 3.82% bound-mass difference. The default solver
is unchanged. Neither roundoff agreement nor finite-grid stability establishes
simulation calibration. Before scientific use, refine all influential axes
together at the actual host/particle parameters and inspect the relevant
observable, including threshold/discontinuity sensitivity.

The first high-order EPS run exposed a removable variance-gap singularity.
Its warning/failure remains preserved. The selected-domain finite-limit fix
was tested independently; new complete catalogs at orders 64 and 200 contain
no warnings and match the prior successful values bitwise. No clipping,
absolute-value correction or new normalization was introduced.

Prompt-cusp scientific validation remains deferred because the required
input spectra are unavailable, as explicitly agreed with the user. The
installed API raises the documented informative missing-data error; no
substitute spectrum was used. Existing CDM boost, MC and threshold tests
are separate observable contracts.
