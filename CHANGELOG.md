# Changelog

## 2.0.0rc1 — migration review candidate

The standard `sashimi_c` import now returns named weighted catalogs through
shared staged execution. Product `physics_mode`/legacy execution is removed;
independent frozen A/B references preserve historical calculations. Canonical
units, stable NFW inversion and finite EPS normalization are explicit.

Mass functions, cumulative counts, mass fractions, annihilation boosts and
seeded realizations remain available. Boost tables have a portable input manifest;
incomplete tables require explicit, warned approximation. Prompt-cusp calculations
remain optional and require the documented input data. `pert2_shanks` and
`ct_th=0` defaults remain fixed; Picard is an explicit alternative.

This candidate supports Python 3.11–3.13 and depends on
`sashimi-itamae>=0.2.0rc1,<0.3`. Candidate wheels are supplied locally;
public index availability, main integration and publication remain separate
post-review operations. No tolerance enlargement or old-fixture overwrite is
part of release preparation.
