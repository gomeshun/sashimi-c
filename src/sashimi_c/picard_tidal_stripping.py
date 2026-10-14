"""CDM coefficient/state adapter to ITAMAE's shared Picard integrator.

The numerical iteration, interpolation, history integration and fallbacks
live in ITAMAE. These compatibility names preserve the existing import path.
"""

import hashlib

import numpy as np
from itamae.evolution import PicardDomain, PicardMassLossSolver, PicardMassLossTable


def physics_key(solver):
    """Fingerprint mutable host inputs, following upstream C PR #22.

    Numerical cache fields are excluded. Custom providers with hidden/nested
    state must expose an immutable ``picard_physics_key()`` value covering it.
    This key is process-local, not a serialization or scientific identity.
    """
    def inputs(owner):
        result = []
        for name, value in sorted(getattr(owner, "__dict__", {}).items()):
            if name.startswith(("_picard", "_eps", "eps_")):
                continue
            if isinstance(value, (float, int, str, bool, np.number)):
                result.append((name, repr(value)))
            elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                result.append((name, value.dtype.str, value.shape,
                               hashlib.sha256(value.tobytes()).hexdigest()))
            elif callable(value):
                result.append((name, id(value)))
        return tuple(result)

    result = [inputs(solver)]
    backend = getattr(solver, "itamae_cosmology", None)
    if backend is not None:
        # NativeFlatLCDM is immutable; adapters can also have mutable scalars.
        result.append(("cosmology", id(backend), inputs(backend)))
    if hasattr(solver, "picard_physics_key"):
        result.append(("provider", solver.picard_physics_key()))
    for name in ("Phi", "zetaMz", "Mzvir"):
        value = getattr(solver, name)
        result.append((name, id(getattr(value, "__func__", value))))
    return tuple(result)


def get_picard_solver(solver):
    """Bind CDM-owned coefficients and state identity to the common backend."""
    backend = getattr(solver, '_picard_backend', None)
    if backend is None or (backend.z_min, backend.z_max) != (solver.z_min, solver.z_max):
        backend = PicardMassLossSolver(
            lambda z: solver.Phi(z), lambda z: solver.zetaMz(z), lambda z: solver.Mzvir(z),
            z_min=solver.z_min, z_max=solver.z_max,
            state_key=lambda: physics_key(solver),
            domain=PicardDomain((0., 7.), (-24., 3.)))
        solver._picard_backend = backend
    return backend


class PicardTidalStrippingTable(PicardMassLossTable):
    """Compatibility wrapper for an explicit ITAMAE table of CDM coefficients."""

    def __init__(self, solver, *args, **options):
        super().__init__(get_picard_solver(solver), *args, **options)
