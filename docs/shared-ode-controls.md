# Shared tidal integration controls

The existing tidal differential equation and solver choice remain owned by this
variant. odeint integration, validation, Jacobian handling and failure checks
now use ITAMAE's shared controller. The default 100 requested times, tolerance
defaults and final/multi-output shape are preserved, including repeated times
and zero evolution. A supplied Jacobian retains this method's state-first
signature through a small argument-order adapter.

Pass rtol/atol and the documented SciPy step/order/Jacobian options directly to
the tidal method. args, tfirst and full_output are controlled internally and
cannot be supplied as duplicate overrides. The returned object remains the
mass array; failed integration does not return a diagnostics object as mass.

The public numerical dependency is verified against the exact ITAMAE input in
the workflow and lock. Direct SciPy comparisons of the same tidal RHS and all
existing catalog regressions are used to verify this execution-only migration.
