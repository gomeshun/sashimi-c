# Shared ITAMAE Picard default

The public tidal method now defaults to `picard_table`. The numerical Picard
iteration, endpoint interpolation, full-history integration and DOP853 fallback
are implemented once in `itamae.evolution.PicardMassLossSolver`. This package
supplies only its physical rate, mass exponent, host history, units and state
identity. C and W request endpoints; SI requests complete histories.

The previous `pert2_shanks` calculation remains available by explicitly passing
`method="pert2_shanks"`. Historical golden inputs and archived outputs retain that
selection; they are not regenerated under the new default. Use
`method="dop853"` for a high-accuracy direct log-mass reference. Solver keyword
arguments are forwarded to the selected numerical method.

The Picard path uses three nonlinear updates plus a convergence estimate from
one further update. Endpoint tables default to 48 redshift nodes, 32 mass-ratio
nodes, 129 integration points and cubic interpolation. SI histories default
to 100 integration points and at least 32 mass nodes for interpolated batches,
with at most 0.25 dex spacing. The model supplies the domain `0 <= z <= 7`
and `-24 <= log10(m_acc/M_host(z_acc)) <= 3`.

Outside that domain, or when the iteration estimate exceeds `8e-4`, ITAMAE
uses DOP853 with `rtol=1e-11`, `atol=1e-12`. Fallbacks emit a warning and are
counted in catalog `picard_diagnostics`, together with actual table/history
settings. The iteration check does not guarantee interpolation or quadrature
accuracy; grid-refinement and direct-ODE comparisons are separate validation.

The physical calibration, accretion population, concentration quadrature,
profile prescription, units and survival threshold are unchanged. The physical
`calculation_specification` is retained; the selected numerical method and
backend have a new solver identifier and recorded options. Recompute evolved
results under the new default; never relabel old results as Picard calculations.

This requires the new `sashimi-itamae>=0.2.0rc2` candidate. CI and the development
lock pin the exact core commit. Install the paired candidate artifacts or the
pinned development sources; publication is a separate step.
