# Scientific documentation and test review

## Scope and baseline

Compare SASHIMI changes with the author-maintained `upstream/main`, recording
its exact revision. Follow moved upstream definitions through their original
qualified names; a file move alone does not make an existing function new.
ITAMAE is the new shared library and has no author-upstream baseline: all of
its functions and classes are in scope. When an author-upstream remote is unavailable, record the explicitly
selected owner baseline and do not imply that it is Ando's upstream branch. Include private helpers, nested
functions, protocols, fixtures, build utilities and tests, not only exports.
Preserved historical scripts in validation/reference archives are evidence: do
not rewrite them and invalidate their recorded source hashes. Document any
new active analysis or validation tools instead.

## Functions and classes

Every new function or class must have a NumPy-style docstring. Start with the
physical or numerical purpose and explain how a researcher should use it.
Document parameters, shapes, units, mass/redshift conventions, returned
quantities and relevant exceptions in the standard NumPy sections. Classes
must describe their state and intended role; constructors and methods also
need docstrings. State the governing equation or approximation, integration
direction, assumptions and validated domain when relevant. Explain numerical
controls and their limits: an iteration tolerance is not a total accuracy
bound. Keep inherited upstream equations identifiable.

A technical summary such as "adapter", "validation helper" or "tests the
API" is insufficient. For infrastructure, explain the scientific consequence
(e.g. mixing cosmologies, losing mass units, reusing stale tidal tables, or
misidentifying the source of a published catalogue). Do not invent a physical
interpretation for build or compatibility checks: label their support role.
Use examples when they clarify calling conventions or numerical choices.

## Tests

Every test, fixture and helper must have a NumPy-style docstring. A test's
Notes section must explain the physical/numerical question, the controlled
setup or independent reference, explicit pass and failure conditions, and
what the test cannot establish. Document parametrized inputs and fixtures.
Explain tolerances as numerical accuracy, algebraic identity, serialization
identity or historical regression; do not present a frozen result or green
suite as physical calibration or an empirical validation of the model.

Assess tests individually for scientific necessity, protection against silent
misuse, reproducibility/packaging support, duplication, and excess dependence
on implementation details. Keep a named, source-linked review of proposed
retirements or consolidation, the surviving coverage, and the risk of removal.
Do not delete or weaken tests merely to obtain a passing suite. Candidate
status is a proposal, not authorization to remove a test. Where strictness
concerns only a subset of assertions, identify that subset rather than
proposing removal of useful scientific coverage.

## Attribution and evidence

Do not choose or add a license on behalf of the upstream maintainer. Preserve
upstream provenance without inventing permission grants or copyright claims.
Keep scientific run revisions separate from subsequent documentation-only
commits; verify executable code has not changed before reusing old results.
