# Numerical notes for SASHIMI-C

The original module imports, public classes, tuple order, units and positional
arguments remain supported. See the example notebook for catalog usage and
[tests](../tests/README.md) for installation checks.

## Changes affecting results

The Picard solver evaluates the existing tidal mass-loss equation without
the expansion used by `pert2_shanks`. The host history, stripping coefficients,
EPS prescription and structural evolution relations retain their definitions.
Changing the solver can change final masses, profiles and observables; agreement
with a direct ODE solution is distinct from agreement with the old approximation.

## Tidal stripping solver

The default is `method="picard_table"`. Use `method="pert2_shanks"` to select the
previous default explicitly, or `method="dop853"` for direct log-mass integration.
Other existing named solvers remain available. Unknown options raise errors.

Endpoint tables use 48 accretion-redshift nodes, 32 log-mass-ratio nodes,
129 integration points, cubic interpolation, three nonlinear updates and a fourth
convergence check.

The automatic table envelope is `0 <= z_obs <= z_acc <= 7` and
`-24 <= log10(ma/Mvir(z_acc)) <= 3`. Outside it, or after a failed convergence
check, `PicardFallbackWarning` and `solver._picard_events` record direct-ODE
fallback. Queries do not silently extrapolate; invalid physical input raises.
Caches include the host/background state, final redshift, numerical options and
particle settings where applicable. Wrapper calls rebuild after state changes.

Each variant supplies its own host history, background and stripping coefficients.
The standalone helper retains the MIT notice from SASHIMI-C PR #5, commit
`88ae730762fb153be7a7433bb563b0b8ab3ec2c2`.

## Validation scope

The checked domains met a `1e-3` relative mass-error gate against refined
independent ODE integrations. This is a finite-domain numerical check, not a
bound on changes from older approximations or a physical calibration.
Simulation agreement and downstream inference require separate validation.

The [historical validation report](https://github.com/gomeshun/sashimi-c/blob/c8f80295e86f61dcf39ee3d427217d3ae12099ba/docs/standalone-maintenance.md)
records the evaluated grids, reference methods, timings and limitations and
links to the full development evidence at that fixed commit. Those generated
reports and intermediate arrays are not needed in a working checkout.
