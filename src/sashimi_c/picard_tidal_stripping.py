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

    Parameters
    ----------
    solver : TidalStrippingSolver or object
        C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
        mutable physical state.

    Returns
    -------
    tuple
        Process-local identity of all inspected scalar/array/callback inputs plus an
        optional provider key.

    Notes
    -----
    C supplies the host history, Phi and zeta; ITAMAE integrates d
    ln(m)/dz=Phi(z)*(m/M_host(z))**zeta(z). The declared validation rectangle is z in [0,7]
    and log10(ma/M_host) in [-24,3]. Outside it, guarded solver requests use a counted
    direct-ODE fallback. Iteration tolerance excludes integration/interpolation error.
    """

    def inputs(owner):
        """Fingerprint numerical provider inputs while excluding derived tidal caches.

        Parameters
        ----------
        owner : object
            Solver or cosmology provider whose mutable inputs must invalidate old tidal tables.

        Returns
        -------
        tuple
            Ordered scalar values, array dtype/shape/content hashes and callable identities.

        Notes
        -----
        C supplies the host history, Phi and zeta; ITAMAE integrates d
        ln(m)/dz=Phi(z)*(m/M_host(z))**zeta(z). The declared validation rectangle is z in [0,7]
        and log10(ma/M_host) in [-24,3]. Outside it, guarded solver requests use a counted
        direct-ODE fallback. Iteration tolerance excludes integration/interpolation error.
        """
        result = []
        for name, value in sorted(getattr(owner, "__dict__", {}).items()):
            if name.startswith(("_picard", "_eps", "eps_")):
                continue
            if isinstance(value, (float, int, str, bool, np.number)):
                result.append((name, repr(value)))
            elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                result.append(
                    (
                        name,
                        value.dtype.str,
                        value.shape,
                        hashlib.sha256(value.tobytes()).hexdigest(),
                    )
                )
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
    """Bind CDM-owned coefficients and state identity to the common backend.

    Parameters
    ----------
    solver : TidalStrippingSolver or object
        C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
        mutable physical state.

    Returns
    -------
    PicardMassLossSolver
        Shared integrator bound to this C solver physical callbacks and validation domain.

    Notes
    -----
    C supplies the host history, Phi and zeta; ITAMAE integrates d
    ln(m)/dz=Phi(z)*(m/M_host(z))**zeta(z). The declared validation rectangle is z in [0,7]
    and log10(ma/M_host) in [-24,3]. Outside it, guarded solver requests use a counted
    direct-ODE fallback. Iteration tolerance excludes integration/interpolation error.
    """
    backend = getattr(solver, "_picard_backend", None)
    if backend is None or (backend.z_min, backend.z_max) != (
        solver.z_min,
        solver.z_max,
    ):
        backend = PicardMassLossSolver(
            lambda z: solver.Phi(z),
            lambda z: solver.zetaMz(z),
            lambda z: solver.Mzvir(z),
            z_min=solver.z_min,
            z_max=solver.z_max,
            state_key=lambda: physics_key(solver),
            domain=PicardDomain((0.0, 7.0), (-24.0, 3.0)),
        )
        solver._picard_backend = backend
    return backend


class PicardTidalStrippingTable(PicardMassLossTable):
    """Compatibility wrapper for an explicit ITAMAE table of CDM coefficients.

    Parameters
    ----------
    solver : TidalStrippingSolver or object
        C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
        mutable physical state.
    *args : tuple
        Additional positional parameters forwarded unchanged to the supplied numerical or
        physical callback.
    **options : Mapping, optional
        Explicit numerical controls for the named solver or table. Unsupported or reserved
        keys are rejected by the receiving routine.

    Notes
    -----
    C supplies the host history, Phi and zeta; ITAMAE integrates d
    ln(m)/dz=Phi(z)*(m/M_host(z))**zeta(z). The declared validation rectangle is z in [0,7]
    and log10(ma/M_host) in [-24,3]. Outside it, guarded solver requests use a counted
    direct-ODE fallback. Iteration tolerance excludes integration/interpolation error.
    """

    def __init__(self, solver, *args, **options):
        """Build an explicit shared endpoint table using this C model tidal coefficients.

        Parameters
        ----------
        solver : TidalStrippingSolver or object
            C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
            mutable physical state.
        *args : tuple
            Positional PicardMassLossTable constructor settings after the bound solver.
        **options : dict
            Keyword PicardMassLossTable settings such as target redshift, grid resolution and
            iteration tolerance.

        Notes
        -----
        C supplies the host history, Phi and zeta; ITAMAE integrates d
        ln(m)/dz=Phi(z)*(m/M_host(z))**zeta(z). The declared validation rectangle is z in [0,7]
        and log10(ma/M_host) in [-24,3]. Outside it, guarded solver requests use a counted
        direct-ODE fallback. Iteration tolerance excludes integration/interpolation error.
        """
        super().__init__(get_picard_solver(solver), *args, **options)
