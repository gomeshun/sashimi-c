# Independent pre-change native-API regression records

These eleven small records were generated **before** native-API implementation
from separately materialized, blob-SHA-verified migration sources:

- SASHIMI-C `0aa33a2a0935b3f0c4bf62072b36186ecc9f1047`
- ITAMAE `23d01e8758a88b061b87de9e488c38ec89fd8e4f`

The unmodified sources were built with their exact revision using the supported
build provenance variables, installed in an isolated Python 3.12 environment,
and evaluated with one BLAS thread. Each JSON records full legacy parameters,
source revisions and the NPZ SHA256. NPZ files contain named physical columns
and factorized weights. Tests compare both the legacy and new native paths
against these independent saved values, rather than only comparing two entry
points after a shared refactor. Numerical defaults were not changed to make
these tests pass.

The older independent `../B-all.npz` reference is retained and tested separately.
This small API regression set is not new external scientific calibration.

Cross-platform frozen-reference comparisons use rtol=5e-12 and atol=0, matching
the pre-existing B reference policy; boolean fields are exact. Old/new paths
within one runtime must match exactly. The validation run that produced this
change also observed exact agreement with these saved references.

## Supplemental survival-cut case (2026-10-07)

The original eleven records remain unchanged. The original `survival` case uses
`ct_th=0.77`, for which all 16 nodes pass, so it does not detect an ignored
threshold. `survival_cut` retains the same grid and uses `ct_th=10`: 6 nodes
survive and 10 are excluded. This threshold is a test input, not a new default.

The supplemental record was generated from the same unchanged pre-API C and
ITAMAE revisions listed above. Both sources were independently exported with
`git archive`, built with their exact source-revision overrides, and installed
as wheels in a separate environment. The generator verified the embedded source
identities and called only `SubhaloProperties().subhalo_catalog_calc` with the
parameters in `survival_cut.json`; it saved the returned columns and factorized
weights using `numpy.savez`. The JSON records the environment and NPZ hash.
Both current entry points must match that independent result. The native API
test also checks that the threshold changes survival and final weights while
preserving the base and concentration weights.
