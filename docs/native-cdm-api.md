# Native CDM specifications

`CDM` separates reusable model settings from one population's physical inputs.
All accepted settings, defaults and calibration limits are owned by SASHIMI-C.
ITAMAE still executes the existing `PopulationComponents`/`PopulationPipeline`
and transports the same factorized weighted catalog. There is no second
execution engine, universal settings registry, or required numerical wrapper.

```python
from sashimi_c import CDM

model = CDM().configure(
    accretion={"mass_nodes": 40, "host_history_nodes": 20, "redshift_step": 0.1},
    concentration={"quadrature_nodes": 5, "scatter_dex": 0.128},
    stripping={"solver": "pert2_shanks"},
    disruption={"ct_threshold": 0.0},
)
catalog = model.population(
    host_mass_msun=1e12,
    host_mass_definition="200c",
    host_mass_redshift=0.0,
    redshift=0.5,
    accretion_mass_range_msun=(1e6, 1e10),
    accretion_mass_definition="200c",
    accretion_redshift_range=(0.5, 3.0),
)
```

The result is `WeightedSubhaloCatalog`, **not** a new `Population` facade with
`mass_function` or `boost` methods. Use its existing weighted histogram,
selection, serialization and realization methods. Legacy observables and tuple
entry points remain available without changed defaults.

## Settings and immutable reuse

`configure` returns a new model. Partial group overrides preserve unspecified
values. Caller dictionaries and nested solver options are copied and frozen;
`resolved_settings` exposes a recursively immutable description. Runs create
fresh physics objects and stripping solvers; the specification stores no
host mass, catalog or solver cache. No mutable backend injection is accepted by
the native specification. The existing calibrated Planck backend remains.

| Group | Defaults |
| --- | --- |
| accretion | `prescription="yang2011"`, `model=3`, `mass_nodes=500`, `redshift_step=0.01`, `host_history_nodes=200` |
| concentration | `prescription="correa2015"`, `relation=None`, `scatter_dex=0.128`, `quadrature_nodes=5` |
| stripping | `prescription="cdm"`, `solver="pert2_shanks"`, `interpolation_nodes=64`, `profile_change=True`, `solver_options={}` |
| disruption | `prescription="truncation"`, `ct_threshold=0.0` |

Accretion models 1, 2 and 3 retain the existing EPS/modified-EPS/Yang choices.
Solver names remain `pert0`, `pert1`, `pert2`, `pert2_shanks`, `pert3`, `odeint`
and `picard_table`. These change the numerical solver, not the stripping law.
The Picard table retains its existing fixed settings, recorded in the catalog.

Only `odeint` accepts native `solver_options`: scalar `rtol`, `atol`, `h0`,
`hmax`, `hmin`, and integer `mxstep`, `mxhnil`, `mxordn`, `mxords`. An empty mapping
clears previous options; a nonempty mapping partially overrides them. Switching
away from `odeint` requires clearing its options. Callbacks, array tolerances,
critical-time arrays and reserved controller arguments remain available only
through appropriate legacy interfaces, not this immutable native specification.
Positive `h0` is incompatible with decreasing-redshift integration; ODE maximum
orders must be positive, and a positive `hmax` must be at least `hmin`.
Unknown or incompatible settings fail before calculation. The auxiliary
accretion-redshift inversion uses 1000 points and ODE output uses 100 points;
these fixed controls are recorded separately from host inversion.

## Physical domains and reference epochs

Both native mass definitions currently support only `200c`. Values are plain
numbers in Msun. The upper accretion mass bound may be `None`, meaning 0.1 times
the resolved host M200c at z=0. Mass quadrature includes both endpoints and has
at least two nodes. These physical limits are independent of `mass_nodes`.

`host_mass_redshift` describes the supplied host mass; `redshift` describes the
output epoch. Nonzero host reference epochs use the historical 1500-node,
5-dex linear inversion once. The native interface rejects nonfinite,
nonmonotonic or out-of-bracket inversions instead of extrapolating. A zero
reference epoch bypasses inversion exactly. The old `M0_at_redshift` entry point
and its historical behavior are unchanged.

An explicit accretion-redshift range is `(lower, upper]`, with lower at least
the output redshift. Nodes start at `lower + redshift_step`. A final node beyond
the requested upper bound is removed, with only floating-point endpoint drift
snapped back to the endpoint. At least two nodes are required. The sampled
last node can therefore be below the upper bound. Quadrature is performed on
the recorded nodes, not on invented boundary nodes.

Omitting the redshift range deliberately retains the historical grid
`arange(redshift + step, 7 + step, step)`. For off-grid settings this can sample
past 7, and the historical stripping interpolation remains bounded by 7.
Use an explicit range to enforce bounded physical support. The metadata records
`legacy-default-grid` versus `explicit-bounded`, requested support, actual nodes,
and actual sampled support. The native finite-support restriction is not
retroactively applied to old APIs. This is an explicit compatibility boundary.

## Experimental concentration replacement

```python
from sashimi_c import TabulatedConcentration

relation = TabulatedConcentration(
    mass_msun=[1e-6, 1e6, 1e12, 1e18],
    redshift=[0.0, 1.0, 8.0],
    c200=[
        [30.0, 20.0, 10.0, 5.0],
        [20.0, 14.0, 7.0, 3.0],
        [5.0, 4.0, 3.0, 2.0],
    ],
)
experimental = model.configure(concentration={"relation": relation})
```

These numbers illustrate the format, **not** a scientifically calibrated table.
`c200` orientation is `(redshift, mass)`. The relation interpolates natural
`log(c200)` bilinearly in linear `z` and `log10(M200c/Msun)`. Data are copied to
immutable tuples. Axes must be finite, strictly increasing, and have at least
two points. Masses and concentration must be positive; redshifts nonnegative.
There is no extrapolation or clipping. The table must cover subhalo **and host**
queries, including stripping interpolation. Preparation checks both grids;
every later evaluation also checks its exact requested coordinate.

A small C-owned `evaluate`/`describe` contract is used by the physics object
and solver. The public specification currently accepts only this supported
immutable table implementation, not arbitrary stateful callbacks. The same
relation is injected into the fresh subhalo model and host stripping solver.
Consequently concentration changes propagate through M200-to-virial conversion,
initial structure, host virial history, stripping coefficients and final mass.
The default Correa branch remains unchanged. For a bounded table with `odeint`,
LSODA integrates forward in `u=-z`, with `dm/du=-dm/dz` and
`tcrit=[-target_redshift]`, so internal steps cannot query below the
table's output-epoch boundary. This control is reserved on that new relation
route and recorded as the effective boundary policy. The no-relation historical
ODE path is unchanged. A supplied redshift step `h0` is sign-transformed consistently; custom
Jacobians are rejected on this bounded relation route. Effective integration
coordinate, `h0` and `tcrit` are recorded.
Bounded table ODE output is additionally checked against
an independent DOP853 integration at zero and nonzero target redshifts.

The relation's payload, units, mass definition, orientation, interpolation rule,
valid domain and content hash are recorded. The model identifier changes with
the table. Results carry an experimental flag: supplying a concentration table
does **not** recalibrate the CDM host-history, variance or tidal prescriptions,
and does not establish external scientific validity.

## Provenance and weights

Catalog provenance contains full effective settings, reference epoch, resolved
host mass and integration ranges, actual redshift nodes, fixed numerical
controls, solver identity, physical specification, source revisions and units.
It remains JSON-compatible and round-trips through `to_npz`/`from_npz`.

Population, concentration and survival weights retain their separate meanings.
The historical tuple weight excludes survival; final catalog weighting includes
it. An API refactor does not renormalize or discard disrupted population nodes.

## Family compatibility boundary

The family migration parent is `gomeshun/sashimi-family` PR #34, head
`7ce00a72f7da4b113261cd255eb85eda0fb6c453`. C's native interface is deliberately
variant-owned. It is not a claim that the same parameter names can be mapped
mechanically to every legacy variant.

| Variant | Contract to preserve in a later native adapter |
| --- | --- |
| C | M200c accretion grid; Planck .315/.674; default `pert2_shanks`, c_t threshold 0 |
| SI | Legacy log mass bounds describe a target-redshift reference M200 grid; actual accretion masses vary by z and formation gates. Paired CDM-reference/SIDM states have separate survival views |
| W | WMAP7 .27/.7; q10 power modification couples sharp-k variance and top-hat concentration. One host-history node activates a deterministic sigmafac branch, not merely a lower quadrature order. Legacy host mass is z=0, default disruption threshold .77 |
| F | Per-accretion-redshift virial-mass grid; m_22 couples transfer, variance/cache and stripping. Initial concentration intentionally retains CDM Correa15; default disruption threshold .77 |

No CDM coordinate, state layout, prescription schema, or default disruption rule
is promoted into ITAMAE. The shared execution/weight/catalog contracts remain
unchanged. SI needs explicit reference-coordinate and paired-state design;
W needs one coupled power/variance/concentration specification; F needs explicit
virial-coordinate and m_22 dependency provenance before native adaptation.
The family z=0 smoke cases alone do not prove semantic equivalence.
