# Standard CDM API and removal of product reproduction paths

The planned breaking release is 2.0.0rc1. `sashimi_c` owns explicit public
classes; the old dynamic `migrate_class` factory and legacy catalog loop are
removed. CDM formulas, tidal solvers and observable reductions are extracted
into `sashimi_c_physics.py`. There is no dependence on a legacy class or hidden
`super()` path. ITAMAE executes the already validated C-owned population stages.

| Previous interface | Current interface and contract |
| --- | --- |
| `sashimi_c_itamae.subhalo_properties` | `sashimi_c.SubhaloProperties` (old import is an alias) |
| `physics_mode=...` | removed; specifying any value raises TypeError |
| tuple calculation | `subhalo_catalog_calc` is primary; `subhalo_properties_calc` converts the same population |
| `halo_model`, `TidalStrippingSolver` | explicit calibrated halo and solver classes; no dynamic migration |
| `subhalo_observables` | `SubhaloObservables`, retaining existing mass/satellite/boost/MC methods |
| historical mode metadata | retained only in old archives; new records identify `calculation_specification` |

Tuple order remains m200_acc, z_acc, r_s_acc, rho_s_acc, m_bound, r_s, rho_s,
c_t, quadrature_weight, survive. Mass is Msun, length Mpc, density Msun/Mpc3;
the tuple weight excludes survival, which remains a separate boolean field.
Velocity-facing historical observable methods use the documented internal
km/s conversion (`threshold_kms * model.km / model.s`). This conversion does
not change the catalog's canonical mass/length/density columns.

The calibrated constants and exact NFW inverse match independently corrected
reference B across every entry of 1,280 nodes. The maximum concentration
relative difference is 1.29e-14 and weight difference 2.17e-15. Historical
consistent observable goldens pass at their unchanged 5e-10 tolerance. The
standard-API suite has 50 passing cases, including tuple conversion, removed
mode rejection, fixed-cosmology growth derivative, target-redshift mass
semantics, survival/profile/weight checks, and explicit Picard/ODE comparisons.
The usage walkthrough (five code cells) and scientific companion (three code
cells) execute in fresh kernels. The plotted axes were inspected. The first
local attempt exposed an editable-install provenance lookup without a module
path; the setup now passes the actual module path, while installed artifacts
continue to use embedded revisions.

Old runtime coexistence assertions were replaced with frozen B comparison and
standard-API assertions; the old tests remain in Git history at 0fce65b.

Component CI pins ITAMAE 5da8dbbbd88f3f45dd7d2fbe66f11203d9632fa7. The candidate
family job enumerates coordinated core/SI overrides and verifies the untouched
base manifest first. Its effective manifest records every tested revision;
these overrides do not update the parent's authoritative compatibility set.

This review unit changes API ownership and provenance. Seeded MC, portable
auxiliary-data handling, larger scientific convergence and final versioned
wheel/sdist validation remain subsequent release-preparation work. Prompt-cusp
scientific calculations are deferred by the user's explicit missing-data
waiver; the informative missing-file boundary is a separate follow-up unit.
