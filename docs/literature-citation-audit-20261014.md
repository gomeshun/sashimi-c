# Literature citation audit: sashimi-c — started 2026-10-14 UTC

Inventoried **238 active definitions**, including inherited/moved functions, private/nested helpers, constructors and tests. Updated **90 docstrings**, of which **59** receive or complete local paper references. The remaining edits clarify units, assumptions or NumPy structure.

The executable baseline before this citation edit is `53523abbd0cb863fb7af1bdf2fd9abe700385970`. Comparison baseline: author upstream/main `e07ef8fa13503bd80818995c0d366311104fb644`. Moving a definition does not make it new; this citation pass also includes inherited definitions.

The audit screens numerical literals and existing attributions, then matches adopted values to primary equations/tables. Generic algorithms, numerical controls, unit conversions and frozen regression outputs do not receive invented physical calibrations. The full [definition index and hash evidence](literature-citation-audit-20261014.json) distinguish edited definitions, unchanged references and archived evidence. An inventory entry is not proof of physical calibration.

## Findings and limits

- Prompt-cusp a_kd=5.3e-12 is retained from the author implementation. The cited papers establish the thermal benchmark, but the exact numerical provenance of this scale factor was not independently recovered. Its source remains unresolved, explicitly in the class and constructor docstrings.
- Correa concentration/history and Ludlow variance fits retain their original calibrations while the adopted background uses rounded Planck-2018 parameters. Misleading descriptions of a new Planck-2018 calibration were removed.
- The concentration scatter 0.128 dex is the z=0, 1e9-1e10 Msun value in Ishiyama et al.; using it across mass/redshift is an assumption. C defaults to ct_th=0, so the literature threshold 0.77 is not falsely described as its active default.
- The historical prompt-cusp calculation assumes a nonempty collapsed sample. This documentation audit does not repair its empty-sample path.

## Verification

The dependency/CI lock advances from ITAMAE `c452c5bbbbf8276870a8ff1f61f77d8bff0d8d7c` to its citation-only revision `4aeb13ed4e3e3e87171fe800e08f74b95c1244a2`. Its executable AST is unchanged.

- All 90 changed docstrings parse as NumPy style, cover explicit arguments and contain their assigned paper identifiers.
- All 44 inventoried tracked Python files have identical executable ASTs after removing only docstrings; assertions, tolerances and equations are unchanged. 0 historical Python files remain byte-identical.
- 116 distinct existing local pytest cases pass on Python 3.11.15. This is source-checkout testing, not a new scientific catalogue or a new calibration.
- The 10 manifest cases initially skipped without FAMILY_TEST_ROOT were rerun with the explicit family checkout and passed; they are included once in the 116-case count.
- The repository CI Ruff scope passes.
- Historical numerical results keep their original calculation revisions. The docstring edit changes source bytes and build identity; executable equivalence does not permit relabeling old outputs. No tests are deleted or weakened.

## Changed definitions and paper mapping

| Definition | References / change |
| --- | --- |
| [`SubhaloProperties.Ffunc`](../src/sashimi_c/__init__.py#L274) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.Gfunc`](../src/sashimi_c/__init__.py#L313) | [parkinson](https://arxiv.org/abs/0708.1382), [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.Ffunc_Yang`](../src/sashimi_c/__init__.py#L359) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties._normalized_yang_kernel`](../src/sashimi_c/__init__.py#L407) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.Na_calc`](../src/sashimi_c/__init__.py#L479) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.subhalo_catalog_calc`](../src/sashimi_c/__init__.py#L629) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`SubhaloProperties.subhalo_properties_calc`](../src/sashimi_c/__init__.py#L1015) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`SubhaloObservables.__init__`](../src/sashimi_c/__init__.py#L1155) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691), [scatter](https://arxiv.org/abs/1101.2020), [survival](https://arxiv.org/abs/astro-ph/0203004), [prompt](https://arxiv.org/abs/2601.19863) |
| [`NFWInitialStructure.initialize`](../src/sashimi_c/_itamae_components.py#L148) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |
| [`TidalProfileEvolution.evolve`](../src/sashimi_c/_itamae_components.py#L237) | [tracks](https://arxiv.org/abs/1002.3376), [structure](https://arxiv.org/abs/1803.07691), [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`TruncationThresholdSurvival`](../src/sashimi_c/_itamae_components.py#L378) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`TruncationThresholdSurvival.select`](../src/sashimi_c/_itamae_components.py#L415) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`CDMUnits`](../src/sashimi_c/_physics.py#L33) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMUnits.__init__`](../src/sashimi_c/_physics.py#L44) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics`](../src/sashimi_c/_physics.py#L67) | [planck](https://arxiv.org/abs/1807.06209), [prompt](https://arxiv.org/abs/2601.19863) |
| [`CDMPhysics.__init__`](../src/sashimi_c/_physics.py#L115) | [planck](https://arxiv.org/abs/1807.06209), [prompt](https://arxiv.org/abs/2601.19863) |
| [`CDMPhysics._synchronize_physics_constants`](../src/sashimi_c/_physics.py#L197) | [planck](https://arxiv.org/abs/1807.06209) |
| [`CDMPhysics._validate_migration_cosmology`](../src/sashimi_c/_physics.py#L227) | [planck](https://arxiv.org/abs/1807.06209) |
| [`CDMPhysics.Hubble`](../src/sashimi_c/_physics.py#L271) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.rhocrit`](../src/sashimi_c/_physics.py#L294) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.growthD`](../src/sashimi_c/_physics.py#L319) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`CDMPhysics.g`](../src/sashimi_c/_physics.py#L352) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.dDdz`](../src/sashimi_c/_physics.py#L375) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`CDMPhysics.dDdz.dOdz`](../src/sashimi_c/_physics.py#L407) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.sigmaMz`](../src/sashimi_c/_physics.py#L459) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.dsdm`](../src/sashimi_c/_physics.py#L488) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.sigma_Ludlow`](../src/sashimi_c/_physics.py#L517) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.sigma_Ludlow.xi`](../src/sashimi_c/_physics.py#L549) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.dsdm_Ludlow`](../src/sashimi_c/_physics.py#L588) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.dsdm_Ludlow.xi`](../src/sashimi_c/_physics.py#L620) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.deltac_func`](../src/sashimi_c/_physics.py#L662) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.s_func`](../src/sashimi_c/_physics.py#L695) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMPhysics.fc`](../src/sashimi_c/_physics.py#L718) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`CDMPhysics.Delc`](../src/sashimi_c/_physics.py#L749) | [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`CDMPhysics.conc200`](../src/sashimi_c/_physics.py#L780) | [correa](https://arxiv.org/abs/1502.00391) |
| [`CDMPhysics.Mvir_from_M200`](../src/sashimi_c/_physics.py#L830) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`CDMPhysics.Mvir_from_M200_fit`](../src/sashimi_c/_physics.py#L880) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`CDMPhysics.Mvir_from_M200_fit.ffunc`](../src/sashimi_c/_physics.py#L920) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`CDMPhysics.Mvir_from_M200_fit.xfunc`](../src/sashimi_c/_physics.py#L952) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`CDMPhysics.Mzi`](../src/sashimi_c/_physics.py#L998) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.Mzzi`](../src/sashimi_c/_physics.py#L1045) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMPhysics.dMdz`](../src/sashimi_c/_physics.py#L1099) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`CDMTidalKernels`](../src/sashimi_c/_physics.py#L1157) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.__init__`](../src/sashimi_c/_physics.py#L1184) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.M0`](../src/sashimi_c/_physics.py#L1238) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.reset_interpolation`](../src/sashimi_c/_physics.py#L1259) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.Mzvir`](../src/sashimi_c/_physics.py#L1309) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.AMz`](../src/sashimi_c/_physics.py#L1334) | [tidal](https://arxiv.org/abs/1803.07691) |
| [`CDMTidalKernels.zetaMz`](../src/sashimi_c/_physics.py#L1371) | [tidal](https://arxiv.org/abs/1803.07691) |
| [`CDMTidalKernels.tdynz`](../src/sashimi_c/_physics.py#L1407) | [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`CDMTidalKernels._get_picard_table`](../src/sashimi_c/_physics.py#L1450) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped_picard_table`](../src/sashimi_c/_physics.py#L1482) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.msolve`](../src/sashimi_c/_physics.py#L1515) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped_odeint`](../src/sashimi_c/_physics.py#L1546) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped_pert0`](../src/sashimi_c/_physics.py#L1812) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped_pert1`](../src/sashimi_c/_physics.py#L1843) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped_pert2`](../src/sashimi_c/_physics.py#L1878) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped_pert3`](../src/sashimi_c/_physics.py#L1923) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMTidalKernels.subhalo_mass_stripped`](../src/sashimi_c/_physics.py#L1976) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels`](../src/sashimi_c/_physics.py#L2028) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.mass_function`](../src/sashimi_c/_physics.py#L2039) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.Nsat_Mpeak`](../src/sashimi_c/_physics.py#L2076) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.Nsat_Vpeak`](../src/sashimi_c/_physics.py#L2121) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.mass_fraction`](../src/sashimi_c/_physics.py#L2166) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.annihilation_boost_factor`](../src/sashimi_c/_physics.py#L2197) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.annihilation_boost_factor_prompt_cusps`](../src/sashimi_c/_physics.py#L2310) | Units, calibration wording, or NumPy structure clarified. |
| [`CDMObservableKernels.subhalo_catalog_MC`](../src/sashimi_c/_physics.py#L2477) | Units, calibration wording, or NumPy structure clarified. |
| [`main`](../src/sashimi_c/boost_iteration_prompt_cusps.py#L6) | [prompt](https://arxiv.org/abs/2601.19863) |
| [`build_ps_interpolators`](../src/sashimi_c/prompt_cusps.py#L30) | [prompt](https://arxiv.org/abs/2601.19863) |
| [`build_ps_interpolators.sigma_calc`](../src/sashimi_c/prompt_cusps.py#L72) | [prompt](https://arxiv.org/abs/2601.19863) |
| [`build_ps_interpolators.dsdm_calc`](../src/sashimi_c/prompt_cusps.py#L115) | [prompt](https://arxiv.org/abs/2601.19863) |
| [`prompt_cusps`](../src/sashimi_c/prompt_cusps.py#L166) | [prompt](https://arxiv.org/abs/2601.19863), [delos](https://arxiv.org/abs/2209.11237), [stucker](https://arxiv.org/abs/2301.04670) |
| [`prompt_cusps.__init__`](../src/sashimi_c/prompt_cusps.py#L212) | [prompt](https://arxiv.org/abs/2601.19863), [delos](https://arxiv.org/abs/2209.11237), [stucker](https://arxiv.org/abs/2301.04670) |
| [`prompt_cusps.powerspectrum`](../src/sashimi_c/prompt_cusps.py#L256) | [prompt](https://arxiv.org/abs/2601.19863) |
| [`prompt_cusps.sigma_function`](../src/sashimi_c/prompt_cusps.py#L299) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143) |
| [`prompt_cusps.cusps_Monte_Carlo`](../src/sashimi_c/prompt_cusps.py#L353) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143), [ellipsoid](https://arxiv.org/abs/astro-ph/9907024) |
| [`prompt_cusps.cusps_Monte_Carlo.f_x`](../src/sashimi_c/prompt_cusps.py#L410) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143) |
| [`prompt_cusps.cusps_Monte_Carlo.d2n_dnudx`](../src/sashimi_c/prompt_cusps.py#L451) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143) |
| [`prompt_cusps.cusps_Monte_Carlo.draw_u`](../src/sashimi_c/prompt_cusps.py#L526) | [prompt](https://arxiv.org/abs/2601.19863), [ellipsoid](https://arxiv.org/abs/astro-ph/9907024) |
| [`prompt_cusps.cusps_Monte_Carlo.fec_fun`](../src/sashimi_c/prompt_cusps.py#L599) | [prompt](https://arxiv.org/abs/2601.19863), [ellipsoid](https://arxiv.org/abs/astro-ph/9907024) |
| [`prompt_cusps._load_or_generate_prompt_cusp_mc`](../src/sashimi_c/prompt_cusps.py#L660) | [prompt](https://arxiv.org/abs/2601.19863) |
| [`prompt_cusps.cusp_properties`](../src/sashimi_c/prompt_cusps.py#L741) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143), [delos](https://arxiv.org/abs/2209.11237), [stucker](https://arxiv.org/abs/2301.04670) |
| [`prompt_cusps.cusp_properties.f_x`](../src/sashimi_c/prompt_cusps.py#L821) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143) |
| [`prompt_cusps.cusp_properties.ddn`](../src/sashimi_c/prompt_cusps.py#L858) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143) |
| [`prompt_cusps.cusp_properties.n_peaks`](../src/sashimi_c/prompt_cusps.py#L898) | [prompt](https://arxiv.org/abs/2601.19863), [bbks](https://doi.org/10.1086/164143) |
| [`test_normalized_yang_kernel_against_independent_definite_integral`](../tests/test_eps_normalized_limit.py#L11) | [yang](https://arxiv.org/abs/1104.1757) |
| [`test_cdm_stages_preserve_node_identity_and_initial_enclosed_mass`](../tests/test_itamae_components.py#L42) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691) |
| [`test_itamae_concentration_quadrature_is_normalized`](../tests/test_itamae_migration.py#L69) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691) |
| [`test_migration_rejects_mixed_cosmology`](../tests/test_itamae_migration.py#L164) | [planck](https://arxiv.org/abs/1807.06209) |
| [`test_catalog_metadata_records_mode_solver_weights_and_threshold`](../tests/test_itamae_migration.py#L390) | [planck](https://arxiv.org/abs/1807.06209) |

## Bibliography

- **bbks**: [Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.](https://doi.org/10.1086/164143)
- **conversion**: [Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.](https://arxiv.org/abs/astro-ph/0203169)
- **correa**: [Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.](https://arxiv.org/abs/1502.00391)
- **delos**: [Delos & White (2023), Prompt cusps and the dark matter annihilation signal.](https://arxiv.org/abs/2209.11237)
- **ellipsoid**: [Sheth, Mo & Tormen (2001), Ellipsoidal collapse and an improved model for the number and spatial distribution of dark matter haloes, Eq. (3), Appendix A.](https://arxiv.org/abs/astro-ph/9907024)
- **growth**: [Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5.](https://doi.org/10.1146/annurev.aa.30.090192.002435)
- **ludlow**: [Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Appendix C.](https://arxiv.org/abs/1601.02624)
- **nfw**: [Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.](https://arxiv.org/abs/astro-ph/9611107)
- **parkinson**: [Parkinson, Cole & Helly (2008), Generating Dark Matter Halo Merger Trees, Section 3.1.](https://arxiv.org/abs/0708.1382)
- **planck**: [Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.](https://arxiv.org/abs/1807.06209)
- **prompt**: [Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.](https://arxiv.org/abs/2601.19863)
- **scatter**: [Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.](https://arxiv.org/abs/1101.2020)
- **structure**: [Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).](https://arxiv.org/abs/1803.07691)
- **stucker**: [Stucker et al. (2023), The effect of stellar encounters on the dark matter annihilation signal from prompt cusps.](https://arxiv.org/abs/2301.04670)
- **survival**: [Hayashi et al. (2003), The Structural Evolution of Substructure.](https://arxiv.org/abs/astro-ph/0203004)
- **tidal**: [Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Eqs. (23)-(24).](https://arxiv.org/abs/1803.07691)
- **time**: [Jiang & van den Bosch (2016), Statistics of Dark Matter Substructure: I. Model and Universal Fitting Functions, Eq. (2).](https://arxiv.org/abs/1403.6827)
- **tracks**: [Penarrubia et al. (2010), The impact of dark matter cusps and cores on the satellite galaxy population around spiral galaxies, Eq. (8), Fig. 6.](https://arxiv.org/abs/1002.3376)
- **virial**: [Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).](https://arxiv.org/abs/astro-ph/9710107)
- **yang**: [Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).](https://arxiv.org/abs/1104.1757)
