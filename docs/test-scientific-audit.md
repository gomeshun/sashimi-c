# Scientific test review: sashimi-c — 2026-10-14

Reviewed **72 test definitions** individually. This count is not the number of parametrized cases. The review proposes future changes; all test bodies, assertions and tolerances are retained.

Executable-code baseline before this documentation change: `bc6667571021e664ad5dab1eb1a3f84ad2891760`.
Comparison baseline for added definitions: author upstream/main, `e07ef8fa13503bd80818995c0d366311104fb644`.

## Decisions

| Decision | Definitions |
| --- | ---: |
| Retain | 53 |
| Consolidate without losing conditions | 14 |
| Relax only incidental assertions | 5 |
| Retire or replace, conditionally | 0 |

A consolidation retains every independent scenario and positive/negative control. A relaxation removes an implementation constraint only after preserving the physical or provenance invariant. A retirement is conditional on the replacement/surviving coverage stated below; no candidate is an instruction to delete a test.

Exact equality remains appropriate for lossless serialization, unchanged input identity, deterministic weight factorization and algebraic contracts. A near-roundoff comparison with a frozen fixture protects reproduction of a specified calculation; it is not an independent physical accuracy requirement. Analytic solutions, independent quadrature, unit identities and controlled refinement have a separate scientific role.

## Validation of this documentation change

- NumPy docstrings parsed for 221 in-scope definitions; all explicit parameters are documented.
- All 44 tracked Python files have identical executable ASTs after removing only module/class/function docstrings. The test bodies, decorators and assertions are unchanged.
- Preserved reference and historical analysis files remain byte-for-byte unchanged.
- Python 3.11.15: 106 existing pytest cases passed, 0 failures/errors and 0 skips.
- 10 existing family-manifest unittest cases passed with FAMILY_TEST_ROOT explicitly supplied.
- Ruff checks pass; ITAMAE additionally passes its full format and mypy checks.
- The saved numerical sweeps/catalogues retain their original calculation revisions. This audit does not relabel them as runs of a later documentation commit.

## Candidate index

| Test | Proposal | Reason and retained coverage |
| --- | --- | --- |
| [FamilyManifestTests.test_coordinated_overrides_are_immutable_and_unambiguous](../scripts/tests/test_family_manifest.py#L183) | Consolidate without losing conditions | Retain this boundary; equivalent per-package cases can be consolidated across the duplicated C/F manifest helpers. |
| [FamilyManifestTests.test_c_candidate_keeps_f_and_other_sibling_revisions](../scripts/tests/test_family_manifest.py#L223) | Consolidate without losing conditions | Consolidate with the generic candidate-isolation test as a package parameter, preserving coverage that the last-entry F case is not special. |
| [test_archive_rejects_conflicting_explicit_revision](../tests/test_build_source_identity.py#L213) | Consolidate without losing conditions | Consolidate with matching-override and malformed-override cases into one parametrized provenance test; preserve all conflict scenarios. |
| [test_exact_checkout_rejects_conflicting_explicit_revision](../tests/test_build_source_identity.py#L242) | Consolidate without losing conditions | Consolidate override cases while retaining both checkout and archive origins; they protect different identity sources. |
| [test_malformed_explicit_revision_does_not_silently_fall_back](../tests/test_build_source_identity.py#L270) | Consolidate without losing conditions | Consolidate malformed values into the common override matrix; rejecting ambiguous provenance remains required. |
| [test_matching_explicit_archive_revision_is_allowed](../tests/test_build_source_identity.py#L300) | Consolidate without losing conditions | Merge this positive control with conflicting-override coverage so the rejection tests cannot pass by rejecting every build. |
| [test_golden_fixture_provenance_is_complete](../tests/test_itamae_migration.py#L35) | Consolidate without losing conditions | Consolidate with historical-fixture identity checks, preserving source and prescription evidence rather than repeated format assertions. |
| [test_itamae_concentration_quadrature_is_normalized](../tests/test_itamae_migration.py#L69) | Consolidate without losing conditions | Move the generic Hermite-moment assertion into ITAMAE or consolidate there; retain variant-level tests that verify these weights actually enter a catalogue. |
| [test_full_catalog_and_observables_match_preserved_consistent_golden](../tests/test_itamae_migration.py#L323) | Consolidate without losing conditions | Overlap with the full-array corrected reference warrants consolidation or a historical regression tier; retain observable coverage that the array comparison lacks. |
| [test_catalog_metadata_records_mode_solver_weights_and_threshold](../tests/test_itamae_migration.py#L367) | Relax only incidental assertions | Keep provenance and round-trip checks; fixed release-version strings and exact identifier spelling may be relaxed in favor of installed version and explicit fields. |
| [test_migration_imports_do_not_monkeypatch_legacy_globals](../tests/test_itamae_migration.py#L530) | Consolidate without losing conditions | Consolidate with the standard-import/tuple behavior test; these assertions alone do not demonstrate general absence of import side effects. |
| [test_historical_reference_remains_explicit_and_shanks_based](../tests/test_picard_migration_sync.py#L35) | Consolidate without losing conditions | Consolidate with golden-provenance coverage while retaining an immutable historical reference identity. |
| [test_public_default_uses_shared_picard](../tests/test_picard_migration_sync.py#L60) | Consolidate without losing conditions | Consolidate with off-grid default-versus-DOP853 coverage, retaining one direct default-selection assertion. |
| [test_picard_table_is_explicit_cached_and_invalidated_with_host_mass](../tests/test_picard_migration_sync.py#L81) | Consolidate without losing conditions | Consolidate with the real-host stale-table rejection test; object reuse and fixed internal dimensions are implementation checks rather than separate physical validation. |
| [test_optional_picard_compares_at_full_catalog_level](../tests/test_picard_migration_sync.py#L107) | Relax only incidental assertions | Retain unchanged-input and metadata checks, but the 25-percent agreement with approximate Shanks is not an accuracy criterion and can be retired once direct-ODE catalogue coverage replaces it. |
| [test_optional_picard_compares_at_observable_level](../tests/test_picard_migration_sync.py#L156) | Relax only incidental assertions | Replace the loose Shanks-based bound with an observable convergence/reference study; retain the unchanged-unevolved positive control. |
| [test_unchanged_host_reuses_table_and_preserves_analytic_mass](../tests/test_picard_state_identity.py#L101) | Relax only incidental assertions | Retain the analytic value; cache-object identity and repeated default-size assertions can be consolidated with configuration tests. |
| [test_vectorized_lookup_and_range_checks](../tests/test_picard_tidal_stripping.py#L202) | Consolidate without losing conditions | Core tests cover generic shape/range behavior, but one CDM adapter boundary case remains useful; consolidate duplicate cases rather than remove all coverage. |
| [test_standard_import_and_tuple_conversion_use_the_same_population](../tests/test_standard_api.py#L100) | Relax only incidental assertions | Keep result equivalence; the assertion banning 'Migration' in class ancestry is incidental and a candidate for removal. |

## Individual assessment of every test

### scripts/tests/test_family_manifest.py

#### 1. `FamilyManifestTests.test_candidate_only_replaces_one_revision_and_records_base`

[Source](../scripts/tests/test_family_manifest.py#L132) · **Retain** · Role: reproducibility.

**Purpose.** Keep a candidate family calculation separate from the recorded compatible release.
**Pass.** Only the candidate revision changes in the effective manifest; siblings, canonical manifest and recorded base remain intact.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because a mixed source set would invalidate the claim about which family was tested.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 2. `FamilyManifestTests.test_coordinated_core_candidate_is_recorded_without_changing_other_siblings`

[Source](../scripts/tests/test_family_manifest.py#L158) · **Retain** · Role: reproducibility.

**Purpose.** Record a coordinated core-and-variant candidate without moving other physical models.
**Pass.** The effective manifest changes ITAMAE and F only and records both overrides while C remains fixed.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain until candidate validation no longer permits coordinated core changes.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 3. `FamilyManifestTests.test_coordinated_overrides_are_immutable_and_unambiguous`

[Source](../scripts/tests/test_family_manifest.py#L183) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Reject candidate sets that cannot identify one reproducible source combination.
**Pass.** Unknown packages, moving refs, conflicting candidate overrides and overrides in promoted mode raise ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain this boundary; equivalent per-package cases can be consolidated across the duplicated C/F manifest helpers.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 4. `FamilyManifestTests.test_promoted_mode_keeps_every_canonical_revision`

[Source](../scripts/tests/test_family_manifest.py#L207) · **Retain** · Role: reproducibility.

**Purpose.** Validate the exact recorded family when no candidate override is intended.
**Pass.** Promoted mode reproduces the complete canonical manifest unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain a positive control distinguishing promoted-family evidence from candidate evidence.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 5. `FamilyManifestTests.test_c_candidate_keeps_f_and_other_sibling_revisions`

[Source](../scripts/tests/test_family_manifest.py#L223) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Avoid moving sibling physical models during a CDM-only candidate validation.
**Pass.** Only sashimi-c changes; F and every other sibling retain their recorded revisions.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with the generic candidate-isolation test as a package parameter, preserving coverage that the last-entry F case is not special.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 6. `FamilyManifestTests.test_refuses_canonical_output_path`

[Source](../scripts/tests/test_family_manifest.py#L245) · **Retain** · Role: reproducibility.

**Purpose.** Protect the persistent compatibility record from a temporary candidate calculation.
**Pass.** Attempting to write the effective manifest over compatibility.toml raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain to prevent candidate evidence from being mistaken for integrated release evidence.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 7. `FamilyManifestTests.test_rejects_non_immutable_candidate_ref`

[Source](../scripts/tests/test_family_manifest.py#L267) · **Retain** · Role: reproducibility.

**Purpose.** Require a fixed source revision for a reproducible candidate calculation.
**Pass.** Branch names, abbreviated or malformed SHAs are rejected.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain immutable source identity; lowercase spelling is a serialization policy and can be normalized instead if deliberately supported.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 8. `FamilyManifestTests.test_non_candidate_gitlink_mismatch_fails_before_output`

[Source](../scripts/tests/test_family_manifest.py#L287) · **Retain** · Role: reproducibility.

**Purpose.** Detect an inconsistent sibling source before reporting a candidate family as valid.
**Pass.** A noncandidate gitlink differing from the manifest causes validation failure and no output manifest.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because unrelated sibling drift changes the actual co-installed scientific code.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 9. `FamilyManifestTests.test_candidate_base_mismatch_is_not_hidden_by_override`

[Source](../scripts/tests/test_family_manifest.py#L316) · **Retain** · Role: reproducibility.

**Purpose.** Validate the original recorded family before applying a candidate override.
**Pass.** A mismatched candidate base reference fails and writes no effective manifest.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** An override must not conceal a previously inconsistent provenance record.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

#### 10. `FamilyManifestTests.test_wrong_candidate_wheel_provenance_is_rejected`

[Source](../scripts/tests/test_family_manifest.py#L336) · **Retain** · Role: reproducibility.

**Purpose.** Ensure the tested binary package contains the source selected in the family manifest.
**Pass.** A wheel carrying the base SHA fails provenance checking; the candidate SHA passes.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain installed-artifact identity because source-checkout tests alone do not establish what the wheel contains.
**Limit.** These checks identify the tested dependency set; they do not establish physical model accuracy.

### tests/test_boost_inputs.py

#### 11. `test_positive_table_result_is_unchanged_outside_source`

[Source](../tests/test_boost_inputs.py#L125) · **Retain** · Role: reproducibility.

**Purpose.** Preserve annihilation boosts when valid higher-order tables are loaded from an explicit location.
**Pass.** At orders zero and one the result equals the hash-checked historical method; changing the working directory does not alter the tables or their recorded manifest.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain numerical compatibility and input provenance; this does not validate the boost prescription against simulations.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 12. `test_wrong_physical_table_contract_fails`

[Source](../tests/test_boost_inputs.py#L173) · **Retain** · Role: reproducibility.

**Purpose.** Prevent reuse of boost tables with a different mass unit, variant or prescription.
**Pass.** Changing the declared mass unit, calculation specification or variant raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** A numerically readable table can still represent different physical quantities.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 13. `test_table_corruption_fails`

[Source](../tests/test_boost_inputs.py#L206) · **Retain** · Role: reproducibility.

**Purpose.** Reject a boost table modified after its provenance was recorded.
**Pass.** Changing fsh while retaining the old manifest digest raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Finite but altered table values can silently bias annihilation luminosity.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 14. `test_out_of_domain_is_not_silently_replaced_with_zero`

[Source](../tests/test_boost_inputs.py#L233) · **Retain** · Role: reproducibility.

**Purpose.** Avoid treating an unsupported boost-table query as zero substructure luminosity.
**Pass.** Moving queried accretion masses beyond the table domain raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** A missing table value is not a physical prediction of zero boost.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 15. `test_zero_log_table_is_not_silently_replaced_with_zero`

[Source](../tests/test_boost_inputs.py#L261) · **Retain** · Role: reproducibility.

**Purpose.** Reject logarithmic interpolation of a zero table entry.
**Pass.** A hash-consistent but zero fsh table raises a positivity error.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain this distinction between an unavailable logarithm and a true zero physical contribution.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 16. `test_invalid_order_fails_before_file_access`

[Source](../tests/test_boost_inputs.py#L293) · **Retain** · Role: reproducibility.

**Purpose.** Require a defined nonnegative order for the recursive boost hierarchy.
**Pass.** Negative, fractional or Boolean recursion order raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This is a public numerical-control boundary; it can share a table-input validation matrix without dropping cases.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 17. `test_generator_records_reproducible_inputs_and_refuses_overwrite`

[Source](../tests/test_boost_inputs.py#L322) · **Retain** · Role: reproducibility.

**Purpose.** Preserve the physical inputs used to generate reusable boost tables.
**Pass.** The saved manifest records the full configuration and matching file hashes; a repeated write raises FileExistsError and leaves the manifest unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Overwriting a table would break reproducibility of previous boost calculations.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 18. `test_failed_generator_leaves_no_partial_table`

[Source](../tests/test_boost_inputs.py#L369) · **Retain** · Role: reproducibility.

**Purpose.** Do not leave a failed boost calculation looking like a valid reusable dataset.
**Pass.** An invalid lower mass boundary raises ValueError and leaves the destination empty.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain atomic output behavior so later calculations cannot consume partial tables.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 19. `test_explicit_approximation_retains_frozen_results_and_reports_invalid_points`

[Source](../tests/test_boost_inputs.py#L408) · **Retain** · Role: reproducibility.

**Purpose.** Expose the limitations when explicitly replaying the historical incomplete-table approximation.
**Pass.** For zero or out-of-domain tables, opt-in execution warns, matches the frozen method, and records invalid query count and mask identity.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain only while this historical reproduction mode is supported; it is not evidence that the approximation is scientifically adequate.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 20. `test_generator_can_explicitly_preserve_incomplete_hierarchy`

[Source](../tests/test_boost_inputs.py#L461) · **Retain** · Role: reproducibility.

**Purpose.** Keep recursive boost-table incompleteness visible in saved generation diagnostics.
**Pass.** Opt-in order-one generation warns, records four incomplete cells/lookups, and reproduces the historical equality of Bsh_0 and Bsh_1.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain as a historical-mode guard; do not interpret unchanged recursion levels as physical convergence.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 21. `test_generator_preserves_only_the_documented_zero_width_boundary`

[Source](../tests/test_boost_inputs.py#L507) · **Retain** · Role: reproducibility.

**Purpose.** Limit the historical zero-contribution exception to a zero-measure mass interval.
**Pass.** The single documented zero-width cell is zero in fsh and Bsh_0, warns, and is recorded as one degenerate cell.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This prevents the approximation option from concealing other invalid physical populations.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_build_source_identity.py

#### 22. `test_archive_in_unrelated_repository_keeps_embedded_revision`

[Source](../tests/test_build_source_identity.py#L141) · **Retain** · Role: reproducibility.

**Purpose.** Keep a published catalogue attributable to the code inside its source archive.
**Pass.** An exported package inside an unrelated checkout resolves to its embedded source SHA rather than the enclosing repository SHA.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one integration case per distinct build hook; otherwise a valid numerical result may cite the enclosing project's revision.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 23. `test_unidentified_export_rejects_enclosing_repository`

[Source](../tests/test_build_source_identity.py#L165) · **Retain** · Role: reproducibility.

**Purpose.** Reject an export whose scientific source revision cannot be established.
**Pass.** A source export lacking embedded identity raises RuntimeError instead of using the surrounding repository's SHA.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This prevents false reproducibility claims; it is independent of physical accuracy.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 24. `test_exact_source_checkout_uses_own_revision`

[Source](../tests/test_build_source_identity.py#L191) · **Retain** · Role: reproducibility.

**Purpose.** Identify the actual source checkout used to produce a catalogue.
**Pass.** A package at its own Git root resolves to that checkout's exact revision.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an end-to-end provenance case for each packaging system.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 25. `test_archive_rejects_conflicting_explicit_revision`

[Source](../tests/test_build_source_identity.py#L213) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Prevent a build override from relabelling a preserved scientific source archive.
**Pass.** A supplied revision that conflicts with the archive's embedded SHA raises RuntimeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with matching-override and malformed-override cases into one parametrized provenance test; preserve all conflict scenarios.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 26. `test_exact_checkout_rejects_conflicting_explicit_revision`

[Source](../tests/test_build_source_identity.py#L242) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Prevent an environment override from relabelling the checked-out calculation.
**Pass.** A supplied revision different from the checkout's SHA raises RuntimeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate override cases while retaining both checkout and archive origins; they protect different identity sources.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 27. `test_malformed_explicit_revision_does_not_silently_fall_back`

[Source](../tests/test_build_source_identity.py#L270) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Reject an unusable source identifier before a scientific package is built.
**Pass.** Malformed explicit revision values raise RuntimeError even when another source identity is available.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate malformed values into the common override matrix; rejecting ambiguous provenance remains required.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 28. `test_matching_explicit_archive_revision_is_allowed`

[Source](../tests/test_build_source_identity.py#L300) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Permit an explicitly confirmed archive identity without changing its provenance.
**Pass.** An explicit SHA equal to the embedded archive SHA is accepted and recorded unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Merge this positive control with conflicting-override coverage so the rejection tests cannot pass by rejecting every build.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

### tests/test_eps_normalized_limit.py

#### 29. `test_normalized_yang_kernel_against_independent_definite_integral`

[Source](../tests/test_eps_normalized_limit.py#L11) · **Retain** · Role: numerical.

**Purpose.** Recover the finite normalized EPS limit when the collapse-barrier gap vanishes.
**Pass.** For barrier gaps from zero to ten, the normalized Yang kernel agrees with an independent definite integral at the specified floating-point tolerance.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** The zero-gap limit contributes to host-history quadrature and must not be dropped as a numerical singularity.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 30. `test_host_quadrature_refinement_includes_finite_zero_gap_nodes`

[Source](../tests/test_eps_normalized_limit.py#L41) · **Retain** · Role: numerical.

**Purpose.** Keep accretion rates physical as host-history quadrature samples near-degenerate nodes.
**Pass.** The refined host quadrature returns the expected mass-grid shape with finite nonnegative rates under strict floating-point handling.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because increasing numerical resolution must not introduce NaNs or negative expected counts.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 31. `test_invalid_active_variance_is_not_replaced_by_zero`

[Source](../tests/test_eps_normalized_limit.py#L67) · **Retain** · Role: numerical.

**Purpose.** Reject a zero active variance interval rather than erasing its EPS contribution.
**Pass.** Replacing the mass variance by a constant causes each supported active EPS prescription to raise ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Silently assigning zero weight would hide a broken variance model and bias the inferred population.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 32. `test_zero_width_normalization_support_has_no_runtime_warning`

[Source](../tests/test_eps_normalized_limit.py#L106) · **Retain** · Role: numerical.

**Purpose.** Handle genuine zero-support EPS normalization cells without undefined arithmetic.
**Pass.** The tested prescriptions and host-quadrature orders return finite nonnegative rates with correct shape under strict floating-point handling.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** A physically empty support and an invalid active variance interval need different treatment.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_components.py

#### 33. `test_truncation_threshold_survival_preserves_strict_legacy_cut`

[Source](../tests/test_itamae_components.py#L14) · **Retain** · Role: numerical.

**Purpose.** Apply the declared strict disruption cut at c_t=c_threshold.
**Pass.** Values below or equal to 0.1 and NaN are rejected; the next representable value above 0.1 and 0.2 survive.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain boundary behavior because changing > to >= can change threshold-selected counts.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 34. `test_cdm_stages_preserve_node_identity_and_initial_enclosed_mass`

[Source](../tests/test_itamae_components.py#L42) · **Retain** · Role: numerical.

**Purpose.** Transport a known half-mass stripping law without altering accretion-node identity.
**Pass.** Initial NFW enclosed mass matches virial accretion mass, bound mass is halved, fixed profile parameters are preserved, and the recovered truncated mass agrees within 2e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This component-level physical identity distinguishes node transport errors from solver error.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_migration.py

#### 35. `test_golden_fixture_provenance_is_complete`

[Source](../tests/test_itamae_migration.py#L35) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Keep the historical CDM reference tied to its source, cosmology and comparison policy.
**Pass.** The reference records source revisions, physical modes, constructor parameters and its original 5e-10 relative comparison tolerance.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with historical-fixture identity checks, preserving source and prescription evidence rather than repeated format assertions.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 36. `test_itamae_concentration_quadrature_is_normalized`

[Source](../tests/test_itamae_migration.py#L69) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Preserve the abundance and log-concentration mean when averaging concentration scatter.
**Pass.** Five-node quadrature sums to one and recovers log10 of medians 5,10,20 to about 1e-15 absolute accuracy.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Move the generic Hermite-moment assertion into ITAMAE or consolidate there; retain variant-level tests that verify these weights actually enter a catalogue.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 37. `test_catalog_satisfies_mass_profile_and_weight_invariants`

[Source](../tests/test_itamae_migration.py#L96) · **Retain** · Role: numerical.

**Purpose.** Produce a physical CDM catalogue whose mass agrees with its truncated NFW profile.
**Pass.** All fields and nonnegative weights are finite; bound mass does not exceed accretion mass; NFW reconstruction agrees at rtol=3e-13.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** These algebraic and sign invariants remain useful when algorithms or physical prescriptions change.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 38. `test_migration_rejects_mixed_cosmology`

[Source](../tests/test_itamae_migration.py#L150) · **Retain** · Role: numerical.

**Purpose.** Prevent combining the fixed CDM variance fit with an incompatible background.
**Pass.** Backends with Omega_m0=0.30 or h=0.70 are rejected against the supported 0.315/0.674 model.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain until the entire fitted prescription supports those alternative cosmologies.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 39. `test_catalog_rejects_unphysical_inputs`

[Source](../tests/test_itamae_migration.py#L186) · **Retain** · Role: numerical.

**Purpose.** Reject unsupported physical domains and integration grids before building a CDM population.
**Pass.** Invalid mass, epoch, step, support, scatter, quadrature, EPS prescription, disruption threshold or solver selection raises the expected error.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep physical and numerical boundaries; exact message fragments can be relaxed separately from rejection behavior.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 40. `test_full_catalog_and_observables_match_preserved_consistent_golden`

[Source](../tests/test_itamae_migration.py#L323) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Reproduce the historical Shanks calculation for catalogue summaries and public observables.
**Pass.** Node/survivor counts match exactly and saved mass, boost and count summaries agree with the frozen consistent reference to rtol=5e-10.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Overlap with the full-array corrected reference warrants consolidation or a historical regression tier; retain observable coverage that the array comparison lacks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 41. `test_catalog_metadata_records_mode_solver_weights_and_threshold`

[Source](../tests/test_itamae_migration.py#L367) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Save the physical and numerical choices needed to reproduce the CDM population.
**Pass.** Metadata record source, cosmology, units, solver, threshold and weight rules; saved catalogue metadata and arrays round-trip unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep provenance and round-trip checks; fixed release-version strings and exact identifier spelling may be relaxed in favor of installed version and explicit fields.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 42. `test_shanks_vs_ode_diagnostic_does_not_change_defaults`

[Source](../tests/test_itamae_migration.py#L477) · **Retain** · Role: numerical.

**Purpose.** Compare historical stripping approximations without changing later production calculations.
**Pass.** The diagnostic supplies finite Shanks-versus-ODE differences, and a subsequent default catalogue still records Picard with the declared threshold.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because a diagnostic must not mutate the physical calculation selected by later calls.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 43. `test_migration_imports_do_not_monkeypatch_legacy_globals`

[Source](../tests/test_itamae_migration.py#L530) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Keep compatibility imports bound to the canonical CDM classes.
**Pass.** All tested migration and standard names are identical class objects after imports.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with the standard-import/tuple behavior test; these assertions alone do not demonstrate general absence of import side effects.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_variance.py

#### 44. `test_c_variance_adapter_matches_legacy_model`

[Source](../tests/test_itamae_variance.py#L9) · **Retain** · Role: numerical.

**Purpose.** Expose the CDM variance fit through ITAMAE without changing its physical values.
**Pass.** At the three mass/redshift pairs, sigma and dS/dM are exactly forwarded and variance equals sigma squared to rtol=2e-15.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain as an adapter check; agreement with the same underlying fit does not independently validate that fit.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_picard_migration_sync.py

#### 45. `test_historical_reference_remains_explicit_and_shanks_based`

[Source](../tests/test_picard_migration_sync.py#L35) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Prevent the Picard migration from silently replacing the old solver reference.
**Pass.** The frozen source SHA, Shanks method, threshold and historical mode remain unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with golden-provenance coverage while retaining an immutable historical reference identity.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 46. `test_public_default_uses_shared_picard`

[Source](../tests/test_picard_migration_sync.py#L60) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Select Picard when callers omit the CDM stripping method.
**Pass.** Default and explicit picard_table mass arrays are exactly equal.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with off-grid default-versus-DOP853 coverage, retaining one direct default-selection assertion.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 47. `test_picard_table_is_explicit_cached_and_invalidated_with_host_mass`

[Source](../tests/test_picard_migration_sync.py#L81) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Avoid reusing a CDM stripping table after host mass changes.
**Pass.** Unchanged queries reuse a table with documented default settings; doubling M0 yields a different table.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with the real-host stale-table rejection test; object reuse and fixed internal dimensions are implementation checks rather than separate physical validation.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 48. `test_optional_picard_compares_at_full_catalog_level`

[Source](../tests/test_picard_migration_sync.py#L107) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Isolate the stripping-solver change from the accretion population.
**Pass.** Picard and Shanks keep initial states and base/concentration weights within 5e-11; evolved masses are finite, positive and within 25 percent; metadata expose actual Picard settings.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain unchanged-input and metadata checks, but the 25-percent agreement with approximate Shanks is not an accuracy criterion and can be retired once direct-ODE catalogue coverage replaces it.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 49. `test_optional_picard_compares_at_observable_level`

[Source](../tests/test_picard_migration_sync.py#L156) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Measure the solver-only effect on the total bound-mass fraction.
**Pass.** Unevolved mass fractions agree to 5e-11 and evolved fractions remain finite and within 25 percent between Picard and Shanks.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Replace the loose Shanks-based bound with an observable convergence/reference study; retain the unchanged-unevolved positive control.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_picard_state_identity.py

#### 50. `test_unchanged_host_reuses_table_and_preserves_analytic_mass`

[Source](../tests/test_picard_state_identity.py#L101) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Recover exponential mass loss for an unchanged analytic host.
**Pass.** The table gives 1e6*exp(-0.25) to rtol=3e-15 and reuses the configured default table.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain the analytic value; cache-object identity and repeated default-size assertions can be consolidated with configuration tests.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 51. `test_mutation_rejects_retained_table_and_rebuilds_owned_cache`

[Source](../tests/test_picard_state_identity.py#L124) · **Retain** · Role: numerical.

**Purpose.** Invalidate tidal approximations after physical host inputs change.
**Pass.** Changing mass, rate, array content or callback rejects retained tables; rebuilding recovers the analytic mass with updated coefficients.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain all mutation routes because scalar, in-place-array and callable changes are distinct stale-cache risks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 52. `test_real_solver_mass_setter_and_legacy_table_rejection`

[Source](../tests/test_picard_state_identity.py#L165) · **Retain** · Role: numerical.

**Purpose.** Protect actual CDM host changes and tables lacking a verifiable state key.
**Pass.** Changing M0 rejects the old table; a table with its state key removed is also rejected.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep real-model coverage; merge duplicated host-mass invalidation smoke tests into this stronger case.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 53. `test_cache_fields_do_not_change_identity_and_provider_hook_covers_hidden_state`

[Source](../tests/test_picard_state_identity.py#L191) · **Retain** · Role: numerical.

**Purpose.** Separate numerical cache contents from hidden physical provider state.
**Pass.** Adding numerical caches leaves the key stable, while changing a custom provider key invalidates the retained table.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain the provider hook: hidden physical state cannot be inferred safely from visible scalars alone.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 54. `test_cosmology_replacement_and_mutable_scalar_state_are_detected`

[Source](../tests/test_picard_state_identity.py#L218) · **Retain** · Role: numerical.

**Purpose.** Invalidate stripping tables when the cosmological provider or its parameters change.
**Pass.** Both changing a backend scalar and replacing the backend change the physics key.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because the background changes host growth and the stripping rate even if M0 is fixed.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_picard_tidal_stripping.py

#### 55. `test_picard3_matches_log_ode_reference`

[Source](../tests/test_picard_tidal_stripping.py#L127) · **Retain** · Role: numerical.

**Purpose.** Bound CDM stripping error against independent direct log-mass integration.
**Pass.** Across five accretion redshifts and three mass ratios, the refined three-iteration table differs from DOP853 by less than 1e-3.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain a direct reference; this table has different resolution from the production-default tests, so the two scopes must stay explicit.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 56. `test_three_iterations_improve_challenging_high_redshift_case`

[Source](../tests/test_picard_tidal_stripping.py#L161) · **Retain** · Role: numerical.

**Purpose.** Check whether another Picard update improves a difficult CDM history.
**Pass.** At z_acc=7 and mass ratio 1e-12, three iterations improve on two and stay within 1e-3 of DOP853.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep as a controlled convergence example, not a demand for monotonic improvement for every model and interpolation point.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 57. `test_vectorized_lookup_and_range_checks`

[Source](../tests/test_picard_tidal_stripping.py#L202) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Return physical stripped masses only inside an explicit CDM table's support.
**Pass.** Vectorized masses keep shape, remain positive and no larger than accretion mass; out-of-range mass ratios and redshifts raise ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Core tests cover generic shape/range behavior, but one CDM adapter boundary case remains useful; consolidate duplicate cases rather than remove all coverage.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_runtime_inputs.py

#### 58. `test_seeded_mc_does_not_consume_global_random_state`

[Source](../tests/test_runtime_inputs.py#L30) · **Retain** · Role: reproducibility.

**Purpose.** Reproduce a subhalo realization without perturbing other stochastic calculations.
**Pass.** Equal seeds yield identical one-dimensional output arrays above the selected mass cut and leave NumPy's global random state unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain reproducibility and selection checks; this does not test the statistical calibration of the sampling distribution.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 59. `test_mc_empty_population_and_rng_validation`

[Source](../tests/test_runtime_inputs.py#L63) · **Retain** · Role: reproducibility.

**Purpose.** Handle an empty selected population and unambiguous Monte Carlo inputs.
**Pass.** A cut above all subhaloes returns empty arrays; simultaneous seed and generator or NaN threshold raises ValueError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain these legitimate-empty and invalid-request boundaries.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 60. `test_missing_prompt_spectra_raise_explicit_errors`

[Source](../tests/test_runtime_inputs.py#L90) · **Retain** · Role: reproducibility.

**Purpose.** Stop prompt-cusp calculations when the required physical spectra are unavailable.
**Pass.** Missing Planck CAMB and prompt-cusp spectra raise explicit errors naming the required files.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain: substituting an arbitrary spectrum or a zero result would change the physical model.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_shared_picard.py

#### 61. `test_public_default_covers_off_grid_masses_and_redshifts`

[Source](../tests/test_shared_picard.py#L11) · **Retain** · Role: numerical.

**Purpose.** Check the production CDM default away from interpolation knots.
**Pass.** At z_acc=0.03,0.9,3.17,6.9 and ratios 1e-20..1, default masses agree with direct DOP853 to rtol=1e-3 and use the shared solver.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because tests of an explicitly refined table do not certify the public default.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 62. `test_solver_options_are_applied_and_recorded_by_catalog`

[Source](../tests/test_shared_picard.py#L34) · **Retain** · Role: numerical.

**Purpose.** Make a caller's refinement choices visible in the resulting CDM catalogue.
**Pass.** Four iterations and 257 integration nodes appear in both actual table settings and recorded solver options.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain to ensure scientific convergence studies vary the calculation actually executed.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 63. `test_fallback_is_recorded_and_unknown_options_are_rejected`

[Source](../tests/test_shared_picard.py#L68) · **Retain** · Role: numerical.

**Purpose.** Expose a failed Picard convergence check and reject irrelevant solver controls.
**Pass.** A deliberately strict one-iteration query warns and counts one fallback mass; an unsupported mxstep argument raises TypeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain adapter-level forwarding and reporting; generic fallback correctness is checked independently in ITAMAE.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_shared_tidal_ode.py

#### 64. `test_tidal_solver_preserves_direct_scipy_values`

[Source](../tests/test_shared_tidal_ode.py#L28) · **Retain** · Role: numerical.

**Purpose.** Preserve the variant's physical stripping equation when using the common ODE controller.
**Pass.** With default and explicit controls, the variant's returned history or endpoint is exactly equal to direct SciPy odeint on the same equation and grid.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one integration test per distinct adapter; common-controller unit tests cannot detect variant-specific argument or unit errors.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 65. `test_zero_evolution_preserves_initial_masses`

[Source](../tests/test_shared_tidal_ode.py#L58) · **Retain** · Role: numerical.

**Purpose.** Keep the accreted mass unchanged when no cosmic evolution is requested.
**Pass.** Equal accretion and final redshifts return the input masses exactly.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This identity catches output-index and zero-duration handling errors.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 66. `test_failed_integration_does_not_return_mass_values`

[Source](../tests/test_shared_tidal_ode.py#L83) · **Retain** · Role: numerical.

**Purpose.** Prevent failed physical stripping from being reported as a successful mass calculation.
**Pass.** Restricting odeint to mxstep=1 raises RuntimeError rather than returning masses.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain propagation through the variant boundary even though ITAMAE also tests the numerical controller.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 67. `test_state_first_jacobian_adapter_preserves_stiff_solution`

[Source](../tests/test_shared_tidal_ode.py#L107) · **Retain** · Role: numerical.

**Purpose.** Adapt the historical state-first Jacobian without changing stiff evolution.
**Pass.** For two decay rates 1 and 1000, the supplied Jacobian is actually called with correct argument shapes and the result equals direct odeint.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain while the variant exposes historical Jacobian arguments; it exercises a signature conversion absent from the generic core tests.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_standard_api.py

#### 68. `test_corrected_full_catalog_matches_independent_reference_b`

[Source](../tests/test_standard_api.py#L31) · **Retain** · Role: numerical.

**Purpose.** Preserve the corrected CDM calculation on a frozen independently produced reference case.
**Pass.** The reference hash is valid; every tuple array matches at rtol=5e-12 and Boolean masks exactly; metadata record the current prescription.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one full-array regression, but classify 5e-12 as fixed-algorithm reproducibility, not required accuracy of physical predictions.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 69. `test_removed_mode_cannot_select_a_hidden_calculation`

[Source](../tests/test_standard_api.py#L69) · **Retain** · Role: numerical.

**Purpose.** Avoid accepting an obsolete mode that would imply an unrecorded physical prescription.
**Pass.** All public constructors and catalogue creation reject legacy, consistent or hybrid physics_mode arguments.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain while users may pass old modes; this prevents silent reinterpretation of a requested scientific calculation.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 70. `test_standard_import_and_tuple_conversion_use_the_same_population`

[Source](../tests/test_standard_api.py#L100) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Ensure named catalogues and legacy tuples represent the same physical nodes.
**Pass.** Tuple fields, base/concentration weights and survival equal the canonical catalogue; aliases resolve consistently.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep result equivalence; the assertion banning 'Migration' in class ancestry is incidental and a candidate for removal.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 71. `test_cosmology_units_and_growth_derivative_are_consistent`

[Source](../tests/test_standard_api.py#L138) · **Retain** · Role: numerical.

**Purpose.** Use one background density and growth history throughout CDM evolution.
**Pass.** Critical density equals 3H**2/(8*pi*G) and the backend density; dD/dz agrees with a finite difference at 2e-9; D(0)=1.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** These independent identities detect unit and derivative errors that golden catalogues can preserve.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 72. `test_target_redshift_host_mass_keeps_the_recorded_mass_definition`

[Source](../tests/test_standard_api.py#L166) · **Retain** · Role: numerical.

**Purpose.** Interpret a supplied host mass at the requested epoch consistently with its z=0 equivalent.
**Pass.** A catalogue using M0_at_redshift=True equals one using the recorded converted z=0 host mass, while retaining the original input mass in metadata.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because mixing host mass definitions changes every accretion and stripping history.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.
