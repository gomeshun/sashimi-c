<p align="center">
  <img src="assets/logo.svg" alt="SASHIMI-C logo" width="440">
</p>

## Hands-on usage walkthrough

Start with [the executable usage walkthrough](notebooks/usage_walkthrough.ipynb): a bounded CDM calculation through the standard API, catalogue inspection/export, weighted mass functions, and explicit checks. The **Usage walkthrough** CI executes every cell in a fresh kernel and uploads the executed artifact. This migration deliverable is tracked in [sashimi-family #28](https://github.com/gomeshun/sashimi-family/issues/28).

# Semi-Analytical SubHalo Inference ModelIng for CDM (SASHIMI-C)
[![arXiv](https://img.shields.io/badge/arXiv-1803.07691%20-green.svg)](https://arxiv.org/abs/1803.07961)
[![arXiv](https://img.shields.io/badge/arXiv-1903.11427%20-green.svg)](https://arxiv.org/abs/1903.11427)

The codes allow to calculate various subhalo properties efficiently using semi-analytical models for cold dark matter (CDM). The results are well in agreement with those from numerical N-body simulations.

## Authors

- Shin'ichiro Ando
- Nagisa Hiroshima
- Shunichi Horigome 
- Ariane Dekker

Special thanks to Tomoaki Ishiyama, who provided data of cosmological N-body simulations that were used for calibration of model output.

Please send enquiries to Shin'ichiro Ando (s.ando@uva.nl). The migration release targets Python 3.11–3.13. In any case, we cannot help with any technical issues not directly related to the content of SASHIMI (such as installation, sub-packages required, etc.)

## What can we do with SASHIMI?

- SASHIMI provides a full catalog of dark matter subhalos in a host halo with arbitrary mass and redshift, which is calculated with semi-analytical models.
- Each subhalo in this catalog is characterized by its mass and density profile both at accretion and at the redshift of interest, accretion redshift, and effective number (or weight) corresponding to that particular subhalo.
- It can be used to quickly compute the subhalo mass function without making any assumptions such as power-law functional forms, etc. Only power law that we assume here is the one for primordial power spectrum predicted by inflation! Everything else is calculated theoretically.
- SASHIMI is not limited to numerical resolution which is often the most crucial limiting factor for the numerical simulation. One can easily set the minimum halo mass to be a micro solar mass or even lighter!
- SASHIMI is not limited to Poisson shot noise that inevitably accompanies when one has to count subhalos like in the case of numerical simulations.
- One can calculate the annihilation boost factor.

## What are the future developments for SASHIMI?

- Extension to different dark matter models. The case of warm dark matter (WDM) and self-interacting dark matter (SIDM) has finished: https://github.com/shinichiroando/sashimi-w, https://github.com/shinichiroando/sashimi-si
- Including spatial information.
- Including intrinsic variance that accompanies the host halo evolution.
- Application to various primordial power spectra.
- Including baryonic effects.

## Prompt cusps (dark matter annihilation)

The smallest dark matter halos form *prompt cusps* with steep central density profiles that dominate the annihilation signal. This is relevant **only when discussing annihilation**, so all prompt-cusp code lives in a separate module, `prompt_cusps.py`, which is imported lazily and is never loaded for standard (non-annihilation) runs. To switch it on, pass `prompt_cusps=True` when constructing a class:

```python
from sashimi_c import *

sh = subhalo_observables(1.e12, prompt_cusps=True)   # M0/Msun; turns the prompt-cusp path on
Bsh, Bcusp_dressed, Bcusp_naked, luminosity_ratio, Ncusp_dressed, Ncusp_naked \
    = sh.annihilation_boost_factor_prompt_cusps(n=0)
```

With `prompt_cusps=True`, sigma(M) is computed from the CAMB linear matter power spectrum (with a free-streaming cutoff set by `k_fs_Mpc` / `filter` / `alpha`) instead of the Ludlow fit. The higher-order (`n>0`) boost tables are pre-computed with `boost_iteration_prompt_cusps.py`. See Ando et al. (arXiv:2601.19863).

Higher-order boost inputs use an explicit data directory and a validated source/hash
manifest. See [boost table generation and current boundary limits](docs/boost-tables.md).
Incomplete interpolation domains fail by default. The explicitly approved
`allow_incomplete_tables=True` option retains the historical zero approximation
with warnings, invalid-point counts and provenance; it is not a calibrated
low-mass boundary. `n=0` remains available without boost tables.

## Standard ITAMAE-backed API

The reviewed calculation is now available from `sashimi_c`. Runtime
`physics_mode` selection and dynamic legacy-class migration have been removed.
This is a breaking API/numerical change for the planned 2.0.0rc1 release.

```python
from sashimi_c import SubhaloProperties

model = SubhaloProperties()
catalog = model.subhalo_catalog_calc(
    M0=1e12, dz=0.5, zmax=2.0, N_ma=24, N_herm=3, N_hermNa=8,
    logmamin=6.0, logmamax=10.0, method="pert2_shanks", ct_th=0.0,
)
print(catalog.columns.keys(), catalog.weight_final.sum())
```

Catalogs contain named columns, separate population/concentration/survival
weights, canonical units and versioned calculation/source provenance. The
lowercase class names and `sashimi_c_itamae` imports are aliases of these same
classes. `subhalo_properties_calc` only converts this population to the
historical ten-column tuple; it cannot execute old physics.

The calibrated background remains `OmegaM=0.315`, `h=0.674`. Incompatible
backgrounds are rejected because host history, EPS and concentration are
calibrated together. The default remains `pert2_shanks` with `ct_th=0.0`;
Picard and direct ODE integration remain explicit solver choices.

See the [API migration guide](docs/standard-api-migration.md), the
[usage walkthrough](notebooks/usage_walkthrough.ipynb), and the separate
[scientific comparison](notebooks/scientific_validation.ipynb). Frozen old
versions and individual correction patches are maintained by
[sashimi-family](https://github.com/gomeshun/sashimi-family/tree/codex/migration-release-20260910/validation/references),
outside the product runtime. Historical fixture provenance is retained.

For this unpublished candidate, install the pinned ITAMAE wheel and C wheel
from the review wheelhouse. The package metadata uses a version dependency;
users do not need a Git checkout or VCS dependency. Optional `demo` extras add
notebook/plotting tools; the former `itamae` extra is empty because ITAMAE is now
required. Public release and main integration await peer review.

## Versions

| Version | Description |
|---------|-------------|
| v1.0 | First public release. |
| v1.1 | Improved computational efficiency (perturbative evaluation of tidal stripping). |
| v1.2 | Prompt cusps for the annihilation signal (opt-in via `prompt_cusps=True`, isolated in `prompt_cusps.py`), plus general fixes. |

**Note (v1.2):** the default of `ct_th` (tidal-disruption threshold) changed from `0.77` to `0.0` (no disruption). This affects results for *all* users, not only prompt-cusp runs; pass `ct_th=0.77` to recover the previous behavior. v1.2 also includes a fix to `dDdz` and other minor corrections.

## References

When you use the outcome of this package for your scientific output, please cite the following publications.

- N. Hiroshima, S. Ando, T. Ishiyama, Phys. Rev. D 97, 123002 (2018) [https://arxiv.org/abs/1803.07691]
- S. Ando, T. Ishiyama, N. Hiroshima, Galaxies 7, 68 (2019) [https://arxiv.org/abs/1903.11427]

The SASHIMI codes depend on results from various earlier papers. Listed below are some of the most essential papers. Please make sure to cite them too, if your focus is close to theirs!

- (Concentration-mass-redshift relation) https://arxiv.org/abs/1502.00391
- (Evolution of host halo mass) https://arxiv.org/abs/1409.5228
- (Extended Press-Schechter model) https://arxiv.org/abs/1104.1757
- (Power spectrum and rms mass density) https://arxiv.org/abs/1601.02624
- (Prompt cusps and dark matter annihilation) https://arxiv.org/abs/2601.19863

## Examples

The file 'sashimi_c.py' contains all the variables and functions that are used to compute various subhalo properties. Please read 'sample.ipyb' for more extensive examples.

Here, as a minimal example, is how you generate a semi-analytical catalog of subhalos:

```
from sashimi_c import *

sh = subhalo_properties()  # call the relevant class
M0 = 1.e12*sh.Msun         # input of host halo mass; here 10^{12} solar masses

ma200,z_acc,rs_acc,rhos_acc,m_z0,rs_z0,rhos_z0,ct_z0,weight,survive \
    = sh.subhalo_properties_calc(M0)
```

For inputs and outputs of this function, see its documentation. For reference, it is:

```
-----
Input
-----
M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
    Note that this is *not* the host mass at the given redshift! It can be obtained
    via Mzi(M0,redshift). If you want to give this parameter as the mass at the given
    redshift, then turn 'M0_at_redshift' parameter on (see below).
        
(Optional) redshift:       Redshift of interest. (default: 0)
(Optional) dz:             Grid of redshift of halo accretion. (default 0.1)
(Optional) zmax:           Maximum redshift to start the calculation of evolution from. (default: 7.)
(Optional) N_ma:           Number of logarithmic grid of subhalo mass at accretion defined as M_{200}.
                           (default: 500)
(Optional) sigmalogc:      rms scatter of concentration parameter defined for log_{10}(c).
                           (default: 0.128)
(Optional) N_herm:         Number of grid in Gauss-Hermite quadrature for integral over concentration.
                           (default: 5)
(Optional) logmamin:       Minimum value of subhalo mass at accretion defined as log_{10}(m_{min}/Msun). 
                           (default: -6)
(Optional) logmamax:       Maximum value of subhalo mass at accretion defined as log_{10}(m_{max}/Msun).
                           If None, m_{max}=0.1*M0. (default: None)
(Optional) N_hermNa:       Number of grid in Gauss-Hermite quadrature for integral over host evoluation, 
                           used in Na_calc. (default: 200)
(Optional) Na_model:       Model number of EPS defined in Yang et al. (2011). (default: 3)
(Optional) ct_th:          Threshold value for c_t(=r_t/r_s) parameter, below which a subhalo is assumed to
                           be completely desrupted. Suggested values: 0.77 (disruption) or 0.0
                           (no desruption; default since v1.2).
(Optional) profile_change: Whether we implement the evolution of subhalo density profile through tidal
                           mass loss. (default: True)
(Optional) M0_at_redshift: If True, M0 is regarded as the mass at a given redshift, instead of z=0.
(Optional) method:         Method to calculate the subhalo mass stripping. (default: "pert2_shanks")
                           - "odeint" : use odeint to solve the differential equation.
                           - "pert0" : use perturbative method with zeroth-order correction.
                           - "pert1" : use perturbative method with first-order correction.
                           - "pert2" : use perturbative method with second-order correction.
                           - "pert2_shanks" : use perturbative method with second-order correction 
                             and Shanks transformation.
                           - "pert3" : use perturbative method with third-order correction.
(Optional) kwargs:         Additional arguments for the odeint function.
        
------
Output
------
List of subhalos that are characterized by the following parameters.
ma200:    Mass m_{200} at accretion.
z_acc:    Redshift at accretion.
rs_acc:   Scale radius r_s at accretion.
rhos_acc: Characteristic density \rho_s at accretion.
m_z0:     Mass up to tidal truncation radius at a given redshift.
rs_z0:    Scale radius r_s at a given redshift.
rhos_z0:  Characteristic density \rho_s at a given redshift.
ct_z0:    Tidal truncation radius in units of r_s at a given redshift.
weight:   Effective number of subhalos that are characterized by the same set of the parameters above.
survive:  If that subhalo survive against tidal disruption or not.
```

These outputs are adopted further in various functions of 'subhalo_observables' class. See 'sample.ipynb' for details. They can be used in https://github.com/shinichiroando/dwarf_params to discuss density profiles of dwarf galaxies, as discussed in a related paper: https://arxiv.org/abs/2002.11956

### Tidal-stripping solver compatibility

The historical and migration defaults remain `method="pert2_shanks"`. The
precomputed third-order Picard solver synchronized from `main` is available as
an explicit `method="picard_table"` option. This synchronization does **not**
adopt the proposed default change from PR #5; changing the public scientific
default requires separate PHY-C review.

Historical migration fixtures continue to use repository revision
`9f6713b686805645da459e99522e2049e7dea793`, `method="pert2_shanks"`,
`ct_th=0.0`, and `physics_mode="legacy"` as historical provenance. Reproduction
runs the frozen version outside this product. Current catalog
metadata records the selected stripping method and, for Picard runs, the table
iteration/grid settings.


### Monte Carlo and prompt-cusp inputs

Use `observables.subhalo_catalog_MC(mth, seed=20260910)` or pass an explicit
`numpy.random.Generator` with `rng=`. See [runtime inputs](docs/runtime-inputs.md)
for row-order and reproducibility guarantees. Prompt-cusp calculations accept
`data_dir=` or `SASHIMI_C_DATA_DIR`; missing required spectra raise an informative
`RuntimeError`. The unavailable redshift-31 table and its dependent scientific
validation are explicitly deferred by the project owner.

## Review candidate

The prepared version is `2.0.0rc1`, with a versioned `sashimi-itamae` dependency.
See [release preparation](docs/release-preparation.md), [changelog](CHANGELOG.md)
and [citation metadata](CITATION.cff). The full artifact matrix is recorded in the
family review handoff after verification; no public upload is implied.

## Measured numerical sensitivity

[Resolution and state validation](docs/resolution-and-states.md) records the
full saved-grid evidence, solver differences, finite EPS limits and remaining
use limitations. The last dz refinement (.01 to .005) changes bound mass
fractions by about 0.754% at fixed reduced settings. These are measured
sensitivities, not universal error guarantees or a change of physical defaults.
