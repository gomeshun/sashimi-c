# Picard synchronization comparison

This directory records the bounded small-catalog comparison required by
`sashimi-family#26` when reconciling the reviewed Picard solver from SASHIMI-C
`main` with the ITAMAE migration.

The historical reference remains repository revision
`9f6713b686805645da459e99522e2049e7dea793`, `physics_mode="legacy"`,
`method="pert2_shanks"`, and `ct_th=0.0`. The synchronized Picard source is
`main@e09571be7a1daaef343e97887e34449faf21db7b`. Picard is available only by
explicitly selecting `method="picard_table"`; this synchronization does not
adopt the separate default-switch proposal in PR #5.

`reference.json` is the recorded output of
`scripts/picard_sync_comparison.py` for the bounded teaching grid in that file.
On this grid:

- public legacy versus migrated legacy `pert2_shanks` agrees to
  `4.44e-16` in the maximum relative bound-mass comparison;
- optional Picard versus migrated `pert2_shanks` differs by at most about
  `1.25%` in individual evolved bound masses;
- the evolved total subhalo mass fraction shifts by about `0.86%`;
- the unevolved mass fraction is unchanged.

These numbers are synchronization diagnostics, not a claim that the teaching
grid is scientifically converged or that Picard should become the public
default. The latter remains a separate PHY-C decision.
