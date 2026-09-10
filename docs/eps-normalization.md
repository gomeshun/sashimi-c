# Finite EPS normalization and active mass support

At large host quadrature order, the finite auxiliary-redshift search can give
exactly zero barrier gap on a valid accretion node. Dividing Yang et al. (2011)
Eq. (14) by its normalization then produces 0/0. The previous `nan_to_num`
removed those contributions. The frozen 200-point audit records 4,320 nonfinite
nodes, including 1,369 inside the allowed mass domain, in the parent repository's
`validation/science/sashimi-c/eps-before`.

For barrier gap delta >= 0, variance difference dS > 0 and lower support
variance dSmin > 0, the normalized kernel is

`delta * exp(-delta²/(2*dS)) / (sqrt(2*pi) * dS**1.5 * erf(delta/sqrt(2*dSmin)))`.

Its delta=0 limit is `sqrt(dSmin)/(2*dS**1.5)`. The small-argument evaluation
uses the integral of exp(-x² t²) on [0,1]; tests compare it to independent
adaptive quadrature from zero through finite gaps. Regular arguments retain
the original incomplete-gamma arithmetic. This is an evaluation of the same
normalized equation, without clipping weights or altering the mass prescription.

All three accretion prescriptions evaluate the kernel only on `ma < mmax`.
For prescriptions 1 and 2, a zero lower variance gap makes the normalization
integral diverge. Its reciprocal is zero; that branch is assigned before any
division is evaluated. Negative support gaps, nonpositive active variance gaps
and nonfinite active kernels raise explicit errors. The integrals and support
are variant-owned physics; host quadrature and catalog transport remain shared.

Validation: 14 new tests failed on the corresponding unfixed boundaries (the
first ten before the Yang/support repair, four before the normalization repair).
All 95 package tests pass at their existing reference tolerances. A separate
frozen-method audit records arrays, warnings and changes through host order 200;
its results are candidate scientific evidence, not a replacement for old fixtures.
The solver, thresholds and calibration are unchanged.
