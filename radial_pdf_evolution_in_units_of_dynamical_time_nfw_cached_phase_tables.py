"""radial_pdf_evolution_tdynamical.py

Standalone script (except for imports from sashimi_c) to compute and plot the time
variation of the radial PDF f(r,t) for a set of particles orbiting in an NFW halo.

Workflow
--------
1) Build an NFW halo from (M, z, c) with robust M200c -> Mvir conversion.
2) Sample (E, L, eta, Rc) using the recipe you provided.
3) Precompute per-orbit phase tables (rp, ra, tau(r), Tr, tau0) so r(t) can be
   evaluated quickly for many times.
4) Define a representative dynamical time t_dyn as median(Tr) over the sample.
5) Plot f(r,t) for a grid of dimensionless times tau = t / t_dyn.

Notes
-----
- Only external dependency is `sashimi_c` for: units_and_constants, cosmology, and
  optionally Mvir_from_M200_fit.
- We keep your original helper functions (Phi_NFW, build_halo, sampling, radial_pdf,
  plot_r_pdf, etc.) inside this file for a single-file workflow.

"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm, trange

# ---- sashimi imports (explicit; mirrors prior `from sashimi_c import *`) ----
from sashimi_c import units_and_constants, cosmology

# optional helper; if absent, build_halo falls back to Newton solve
try:
    # prefer direct import if sashimi_c exposes a top-level function
    from sashimi_c import Mvir_from_M200_fit  # type: ignore
except Exception:
    # if not present, try to wrap the method on halo_model
    try:
        from sashimi_c import halo_model
        def Mvir_from_M200_fit(M, z):
            hm = halo_model()
            return hm.Mvir_from_M200_fit(M, z)
    except Exception:
        import warnings
        warnings.warn("Mvir_from_M200_fit not found; using fallback M200c->Mvir solver.")


# =============================================================================
# User-provided halo / sampling helpers (integrated)
# =============================================================================

def fNFW(c):
    r"""NFW helper function.

    Computes :math:`f(c) = \ln(1+c) - c/(1+c)` which appears in the enclosed-mass
    expression of an NFW profile.

    Parameters
    ----------
    c : float or array-like
        Concentration parameter.

    Returns
    -------
    float or ndarray
        Value(s) of f(c).
    """
    return np.log1p(c) - c / (1.0 + c)


def _Delta_vir_BryanNorman(z, cos):
    r"""Virial overdensity :math:`\Delta_\mathrm{vir}(z)` from Bryan & Norman.

    This uses the common fitting formula (flat cosmology) expressed in terms of
    :math:`\Omega_m(z)`.

    Parameters
    ----------
    z : float
        Redshift.
    cos : object
        Cosmology-like object providing ``OmegaM`` and ``g(z)=E(z)^2``.

    Returns
    -------
    float
        Virial overdensity relative to the critical density.
    """
    Om0 = cos.OmegaM
    Ez2 = cos.g(z)
    Omz = Om0 * (1.0 + z) ** 3 / Ez2
    d = Omz - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * d - 39.0 * d * d


def build_halo(M, z, c, mass_def="200c"):
    """Build a self-consistent NFW halo dictionary.

    The returned halo parameters are constructed such that the NFW profile is
    compatible with either:

    - ``mass_def='200c'``: input is (M200c, c200). We compute ``rs`` from these
      and then convert to a virial definition using ``Mvir_from_M200_fit`` if
      available; otherwise a Newton solve is used.
    - ``mass_def='vir'``: input is (Mvir, cvir).

    Parameters
    ----------
    M : float
        Halo mass with the internal unit system used by ``units_and_constants``.
    z : float
        Redshift.
    c : float
        Concentration parameter (c200 or cvir depending on ``mass_def``).
    mass_def : {'200c', 'vir'}
        Definition of the input mass and concentration.

    Returns
    -------
    dict
        Dictionary containing at least ``Mvir, Rvir, cvir, rs, rhos, Vs2`` plus
        auxiliary values (``rhoc, Delta_vir``).
    """
    u = units_and_constants()
    cos = cosmology()

    rhoc = cos.rhocrit(z)
    Delta_vir = _Delta_vir_BryanNorman(z, cos)

    if mass_def == "vir":
        Mvir = float(M / u.Msun) * u.Msun
        cvir = float(c)
        Rvir = (3.0 * Mvir / (4.0 * np.pi * Delta_vir * rhoc)) ** (1.0 / 3.0)
        rs = Rvir / cvir

    elif mass_def == "200c":
        M200 = float(M / u.Msun) * u.Msun
        c200 = float(c)

        # rs from (M200,c200)
        R200 = (3.0 * M200 / (4.0 * np.pi * 200.0 * rhoc)) ** (1.0 / 3.0)
        rs = R200 / c200

        # try preferred conversion if available
        try:
            out = Mvir_from_M200_fit(M200, z)
            try:
                Mvir, Rvir = out
            except (TypeError, ValueError):
                Mvir = out
                Rvir = (3.0 * Mvir / (4.0 * np.pi * Delta_vir * rhoc)) ** (1.0 / 3.0)
            cvir = float(Rvir / rs)
        except NameError:
            # fallback: solve cvir from cvir^3/f(cvir) = (200/Δvir) * c200^3/f(c200)
            target = (200.0 / Delta_vir) * (c200**3 / fNFW(c200))
            cvir = float(c200)
            for _ in range(64):
                f = fNFW(cvir)
                g = cvir**3 / f - target
                # dg/dc = (3c^2 f - c^3 f')/f^2, f' = 1/(1+c) - 1/(1+c)^2
                fp = 1.0 / (1.0 + cvir) - 1.0 / (1.0 + cvir) ** 2
                dg = (3 * cvir**2 * f - cvir**3 * fp) / (f * f)
                step = g / dg
                cvir -= step
                if abs(step) < 1e-10:
                    break
            Rvir = cvir * rs
            Mvir = (4.0 * np.pi / 3.0) * Delta_vir * rhoc * Rvir**3
    else:
        raise ValueError("mass_def must be 'vir' or '200c'.")

    rhos = Mvir / (4.0 * np.pi * rs**3 * fNFW(cvir))
    Vs2 = 4.0 * np.pi * u.G * rhos * (rs**2)

    return dict(
        Mvir=Mvir,
        Rvir=Rvir,
        cvir=cvir,
        rs=rs,
        rhos=rhos,
        Vs2=Vs2,
        rhoc=rhoc,
        Delta_vir=Delta_vir,
    )


def Phi_NFW(r, rs, Vs2):
    r"""NFW gravitational potential with :math:`\Phi(\infty)=0`.

    Parameters
    ----------
    r : float or array-like
        Radius.
    rs : float
        Scale radius.
    Vs2 : float
        Convenience constant defined as ``4π G ρs rs^2`` in this file.

    Returns
    -------
    float or ndarray
        Potential value(s) at ``r``.
    """
    x = np.asarray(r) / rs
    y = np.where(
        np.abs(x) < 1e-6,
        1.0 - x / 2.0 + x**2 / 3.0 - x**3 / 4.0,
        np.log1p(x) / x,
    )
    return -Vs2 * y


def dPhi_dr_NFW(r, rs, Vs2):
    r"""Radial derivative of the NFW potential, :math:`d\Phi/dr`.

    Parameters
    ----------
    r : float or array-like
        Radius.
    rs : float
        Scale radius.
    Vs2 : float
        Convenience constant defined as ``4π G ρs rs^2`` in this file.

    Returns
    -------
    float or ndarray
        ``dPhi/dr`` value(s) at ``r``.
    """
    x = np.asarray(r) / rs
    m = np.log1p(x) - x / (1.0 + x)
    return Vs2 * (m / x**2) / rs


def sample_EL_from_eta_Rc(M, z, c, n=1000, seed=None, mass_def="200c", h=0.7):
    """Sample orbital integrals (E, L) using the paper's recipe.

    This implements the (η, Rc) sampling used elsewhere in the workspace:

    - ``Rc`` is drawn uniformly in ``[0.6, 1.0] * Rvir``.
    - ``η`` is drawn from a Beta distribution whose parameters depend on
      ``Mvir`` and ``z``.
    - ``E`` and ``L`` are then assigned using circular-speed estimates at ``Rc``.

    Parameters
    ----------
    M, z, c : float
        Halo parameters passed to :func:`build_halo`.
    n : int
        Number of samples.
    seed : int or None
        RNG seed.
    mass_def : {'200c', 'vir'}
        Interpretation of ``M`` and ``c``.
    h : float
        Reduced Hubble parameter used in the mass-ratio fit.

    Returns
    -------
    E : ndarray
        Specific orbital energy samples.
    L : ndarray
        Specific angular momentum samples.
    eta : ndarray
        Circularity parameter samples.
    Rc : ndarray
        "Accretion"/initial radius samples.
    halo : dict
        Halo dictionary from :func:`build_halo`.
    """
    u = units_and_constants()
    rng = np.random.default_rng(seed)
    H = build_halo(M, z, c, mass_def=mass_def)

    Rvir = H["Rvir"]
    Mvir = H["Mvir"]
    rs = H["rs"]
    Vs2 = H["Vs2"]

    Rc = rng.uniform(0.6, 1.0, size=n) * Rvir

    log10_Mstar_hinv = 12.42 - 1.56 * z + 0.038 * z * z
    Mstar_hinv = 10.0**log10_Mstar_hinv * u.Msun
    M_hinv = Mvir * h
    ratio = (M_hinv / u.Msun) / (Mstar_hinv / u.Msun)
    C1 = 0.242 * (1.0 + 2.36 * (ratio**0.107))
    eta = rng.beta(2.05, C1 + 1.0, size=n)

    Vc = np.sqrt(u.G * Mvir / Rc)
    E = 0.5 * Vc**2 + Phi_NFW(Rc, rs, Vs2)
    L = eta * Rc * Vc

    return E, L, eta, Rc, H


def sample_EL_infall_shell_at_Rvir(M, z, c, n=1000, seed=None, mass_def="200c", h=0.7):
    r"""Sample (E, L) for an accretion-shell initial condition at ``Rvir(z)``.

    This matches the "all subhalos cross the host virial radius at the same time"
    picture used in [sashimi_c.py](sashimi_c.py): set the initial radius for all
    particles to ``r0 = Rvir(z)`` and draw orbital circularity ``η`` from the same
    Beta distribution recipe.

    We then set the specific energy to the *circular-orbit* energy at ``r0`` and
    the angular momentum to ``L = η L_c(r0)``. For ``η<1``, this implies a nonzero
    radial speed at ``r0``; choosing ``sign_mode='in'`` downstream corresponds to
    inward infall at the virial boundary.

    Parameters
    ----------
    M, z, c : float
        Host-halo parameters passed to :func:`build_halo`.
    n : int
        Number of subhalo orbit samples.
    seed : int or None
        RNG seed.
    mass_def : {'200c', 'vir'}
        Interpretation of ``M`` and ``c``.
    h : float
        Reduced Hubble parameter used in the mass-ratio fit.

    Returns
    -------
    E : ndarray
        Specific orbital energy samples.
    L : ndarray
        Specific angular momentum samples.
    eta : ndarray
        Circularity parameter samples.
    r0 : ndarray
        Initial radius array (all entries are ``Rvir(z)``).
    halo : dict
        Halo dictionary from :func:`build_halo`.
    """
    u = units_and_constants()
    rng = np.random.default_rng(seed)
    halo = build_halo(M, z, c, mass_def=mass_def)

    Rvir = halo["Rvir"]
    Mvir = halo["Mvir"]
    rs = halo["rs"]
    Vs2 = halo["Vs2"]

    r0 = np.full(int(n), float(Rvir), dtype=float) * (Rvir / float(Rvir))

    log10_Mstar_hinv = 12.42 - 1.56 * z + 0.038 * z * z
    Mstar_hinv = 10.0**log10_Mstar_hinv * u.Msun
    M_hinv = Mvir * h
    ratio = (M_hinv / u.Msun) / (Mstar_hinv / u.Msun)
    C1 = 0.242 * (1.0 + 2.36 * (ratio**0.107))
    eta = rng.beta(2.05, C1 + 1.0, size=int(n))

    Vc = np.sqrt(u.G * Mvir / Rvir)
    E0 = 0.5 * Vc**2 + Phi_NFW(Rvir, rs, Vs2)
    L0 = eta * float(Rvir) * float(Vc)

    E = np.full(int(n), float(E0), dtype=float) * (E0 / float(E0))
    L = np.asarray(L0, dtype=float) * (Vc / float(Vc))

    return E, L, eta, r0, halo


def sample_EL_from_eta_Rc_placed_at_Rvir(
    M,
    z,
    c,
    n=1000,
    seed=None,
    mass_def="200c",
    h=0.7,
    *,
    oversample=4,
    max_rounds=50,
):
    r"""Sample (E, L) stochastically, then place all particles at ``r0=Rvir(z)``.

    Intended use
    ------------
    You want orbital invariants to be drawn from the same distribution as
    :func:`sample_EL_from_eta_Rc` (i.e. sample ``Rc`` and ``η`` and set
    ``E = E_circ(Rc)``, ``L = η L_c(Rc)``), *but* the configuration-space initial
    condition should be a delta shell at the host virial radius:

    - sample orbit parameters (E, L) from (Rc, η)
    - set initial radius for every particle to ``r0 = Rvir(z)``
    - keep only orbits that can actually pass through ``r0`` (i.e. ``v_r^2(r0) >= 0``)

    With ``sign_mode='in'`` in :func:`evolve_r_from_cache`, this corresponds to
    subhalos crossing the virial boundary and moving inward at t=0.

    Notes
    -----
    This uses simple rejection sampling. If acceptance is low for your chosen
    distributions, increase ``oversample`` or ``max_rounds``.

    Parameters
    ----------
    M, z, c : float
        Host-halo parameters passed to :func:`build_halo`.
    n : int
        Number of accepted samples to return.
    seed : int or None
        RNG seed.
    mass_def : {'200c', 'vir'}
        Interpretation of ``M`` and ``c``.
    h : float
        Reduced Hubble parameter used in the mass-ratio fit.
    oversample : int
        Factor controlling batch size per rejection-sampling round.
    max_rounds : int
        Maximum number of rejection-sampling rounds.

    Returns
    -------
    E, L, eta, Rc : ndarray
        Accepted orbit samples.
    r0 : ndarray
        Initial radius array, all equal to ``Rvir(z)``.
    halo : dict
        Halo dictionary from :func:`build_halo`.
    """
    u = units_and_constants()
    rng = np.random.default_rng(seed)
    halo = build_halo(M, z, c, mass_def=mass_def)

    Rvir = halo["Rvir"]
    Mvir = halo["Mvir"]
    rs = halo["rs"]
    Vs2 = halo["Vs2"]

    # Precompute eta-distribution parameters (same as sample_EL_from_eta_Rc)
    log10_Mstar_hinv = 12.42 - 1.56 * z + 0.038 * z * z
    Mstar_hinv = 10.0**log10_Mstar_hinv * u.Msun
    M_hinv = Mvir * h
    ratio = (M_hinv / u.Msun) / (Mstar_hinv / u.Msun)
    C1 = 0.242 * (1.0 + 2.36 * (ratio**0.107))

    # rejection sample until we have n that can pass through r0=Rvir
    want = int(n)
    E_acc: list[np.ndarray] = []
    L_acc: list[np.ndarray] = []
    eta_acc: list[np.ndarray] = []
    Rc_acc: list[np.ndarray] = []

    r0_scalar = float(Rvir)
    phi_r0 = float(Phi_NFW(r0_scalar, float(rs), float(Vs2)))

    for _round in range(int(max_rounds)):
        need = want - sum(a.size for a in E_acc)
        if need <= 0:
            break

        m = int(max(need * int(oversample), 256))
        Rc = rng.uniform(0.6, 1.0, size=m) * Rvir
        eta = rng.beta(2.05, C1 + 1.0, size=m)

        Vc_Rc = np.sqrt(u.G * Mvir / Rc)
        E = 0.5 * Vc_Rc**2 + Phi_NFW(Rc, rs, Vs2)
        L = eta * Rc * Vc_Rc

        # pass-through condition at r0: v_r^2(r0) >= 0
        # v_r^2(r0) = 2(E - Phi(r0)) - L^2/r0^2
        vr2_r0 = 2.0 * (np.asarray(E, float) - phi_r0) - (np.asarray(L, float) ** 2) / (r0_scalar**2)
        ok = np.isfinite(vr2_r0) & (vr2_r0 >= 0.0)

        if not np.any(ok):
            continue

        E_ok = np.asarray(E, float)[ok]
        L_ok = np.asarray(L, float)[ok]
        eta_ok = np.asarray(eta, float)[ok]
        Rc_ok = np.asarray(Rc, float)[ok]

        take = min(need, E_ok.size)
        if take <= 0:
            continue

        E_acc.append(E_ok[:take])
        L_acc.append(L_ok[:take])
        eta_acc.append(eta_ok[:take])
        Rc_acc.append(Rc_ok[:take])

    got = sum(a.size for a in E_acc)
    if got < want:
        raise RuntimeError(
            f"Could not draw enough orbits that pass through r0=Rvir. "
            f"Requested n={want}, got {got}. Try increasing oversample/max_rounds."
        )

    E_out = np.concatenate(E_acc)[:want]
    L_out = np.concatenate(L_acc)[:want]
    eta_out = np.concatenate(eta_acc)[:want]
    Rc_out = np.concatenate(Rc_acc)[:want]
    r0_out = np.full(want, r0_scalar, dtype=float) * (Rvir / float(Rvir))

    return E_out, L_out, eta_out, Rc_out, r0_out, halo


def time_averaged_radius_batch_halo(E, L, halo, nr=4096, rmax_factor_Rvir=5.0):
    r"""Compute time-averaged radius and radial period for a batch of orbits.

    For each orbit specified by (E, L) in the given NFW halo, this computes
    the time-average of the radius over one radial period,
    $$\langle r \rangle = \frac{\int r\,dr/|v_r|}{\int dr/|v_r|},$$
    and estimates pericenter/apocenter and the radial period ``Tr``.

    Parameters
    ----------
    E, L : array-like
        Specific energy and angular momentum arrays.
    halo : dict
        Halo dictionary containing at least ``rs, Vs2, Rvir``.
    nr : int
        Number of radial grid points used for numerical quadrature.
    rmax_factor_Rvir : float
        Upper integration limit as a multiple of ``Rvir``.

    Returns
    -------
    r_mean : ndarray
        Time-averaged radius for each orbit.
    rp, ra : ndarray
        Pericenter and apocenter estimates.
    Tr : ndarray
        Radial period estimates.
    """
    rs = halo["rs"]
    Vs2 = halo["Vs2"]
    Rvir = halo["Rvir"]

    # keep units (values are plain floats in SI×unit, so NumPy ops work directly)
    E = np.atleast_1d(np.array(E, dtype=float)) * (E[0] / float(E[0]))
    L = np.atleast_1d(np.array(L, dtype=float)) * (L[0] / float(L[0]))

    rmin = max(1e-8 * rs, 1e-10 * units_and_constants().Mpc)
    rmax = rmax_factor_Rvir * Rvir
    r = np.geomspace(float(rmin / rs), float(rmax / rs), nr) * rs
    dr = np.diff(r)

    phi_r = Phi_NFW(r, rs, Vs2)
    dPhidr = dPhi_dr_NFW(r, rs, Vs2)

    r2 = r[:, None] ** 2
    g = 2.0 * (E[None, :] - phi_r[:, None]) - (L[None, :] ** 2) / r2
    mask = g > 0.0
    any_bound = mask.any(axis=0) & (E < 0.0)

    gpos = np.where(mask, g, np.nan)
    inv_root = np.nan_to_num(1.0 / np.sqrt(gpos), nan=0.0, posinf=0.0, neginf=0.0)
    r_over = np.nan_to_num(r[:, None] * inv_root, nan=0.0)

    I0 = np.sum(0.5 * (inv_root[:-1, :] + inv_root[1:, :]) * dr[:, None], axis=0) * 2.0
    I1 = np.sum(0.5 * (r_over[:-1, :] + r_over[1:, :]) * dr[:, None], axis=0) * 2.0

    r_mean = np.full(E.shape, np.nan, dtype=float) * rs / rs * rs
    good = (I0 > 0) & any_bound
    r_mean[good] = I1[good] / I0[good]

    # circular branch
    need_circ = ~any_bound
    rp = np.full_like(r_mean, np.nan) * rs / rs * rs
    ra = np.full_like(r_mean, np.nan) * rs / rs * rs
    if np.any(need_circ):
        F = (L[None, :] ** 2) - (r[:, None] ** 3) * dPhidr[:, None]
        s = np.sign(F)
        s[s == 0.0] = 1.0
        change = s[:-1, :] * s[1:, :] < 0.0
        cols = np.where(need_circ & change.any(axis=0))[0]
        if cols.size:
            idx = np.argmax(change[:, cols], axis=0)
            r1, r2_ = r[idx], r[idx + 1]
            F1, F2 = F[idx, cols], F[idx + 1, cols]
            with np.errstate(invalid="ignore", divide="ignore"):
                rc = r1 + (0.0 - F1) * (r2_ - r1) / (F2 - F1)
            rc = np.where(np.isfinite(rc), rc, r1)
            r_mean[cols] = rc
            rp[cols] = rc
            ra[cols] = rc

    if np.any(any_bound):
        first_true = np.argmax(mask, axis=0)
        last_true = mask.shape[0] - 1 - np.argmax(mask[::-1, :], axis=0)
        cols2 = np.where(any_bound)[0]
        if cols2.size:
            iL = np.clip(first_true[cols2] - 1, 0, r.size - 2)
            iR = np.clip(last_true[cols2], 0, r.size - 2)
            r1l, r2l = r[iL], r[iL + 1]
            r1r, r2r = r[iR], r[iR + 1]
            g1l, g2l = g[iL, cols2], g[iL + 1, cols2]
            g1r, g2r = g[iR, cols2], g[iR + 1, cols2]
            with np.errstate(invalid="ignore", divide="ignore"):
                rp_est = r1l + (0 - g1l) * (r2l - r1l) / (g2l - g1l)
                ra_est = r1r + (0 - g1r) * (r2r - r1r) / (g2r - g1r)
            rp_est = np.where(np.isfinite(rp_est), rp_est, r1l)
            ra_est = np.where(np.isfinite(ra_est), ra_est, r2r)
            rp[cols2] = rp_est
            ra[cols2] = ra_est

    Tr = np.full_like(r_mean, np.nan) * units_and_constants().s
    Tr[good] = 2.0 * I0[good]
    return r_mean, rp, ra, Tr


def radial_pdf(r, nbins=30, rmin=None, rmax=None):
    """Histogram-based radial PDF estimate.

    Parameters
    ----------
    r : array-like
        Radii (can carry the internal length units).
    nbins : int
        Number of histogram bins.
    rmin, rmax : float or None
        Optional explicit histogram range in kpc units (numeric).

    Returns
    -------
    centers_kpc : ndarray
        Bin centers (with kpc unit attached via the internal unit system).
    pdf_per_kpc : ndarray
        Density-normalized PDF values (per kpc).
    edges_kpc : ndarray
        Bin edges (kpc).
    """
    u = units_and_constants()
    r_num = np.asarray(r, float) / float(u.kpc)
    r_num = r_num[np.isfinite(r_num)]
    if rmin is None:
        rmin = np.nanmin(r_num)
    if rmax is None:
        rmax = np.nanmax(r_num)
    # Handle a degenerate or near-degenerate shell (e.g., tau=0 with all r identical)
    if not (np.isfinite(rmin) and np.isfinite(rmax)):
        raise ValueError("radial_pdf: rmin/rmax are not finite (check input radii).")

    width = float(rmax - rmin)
    if width <= 0.0:
        mid = 0.5 * float(rmin + rmax)
    else:
        mid = 0.5 * float(rmin + rmax)

    # Enforce a minimum width in kpc so numpy can create finite-sized bins.
    min_width_abs = 1e-2  # 0.01 kpc
    min_width_rel = 1e-6 * max(1.0, abs(mid))
    min_width = max(min_width_abs, min_width_rel)

    if width < min_width:
        half = 0.5 * min_width
        rmin = mid - half
        rmax = mid + half
    pdf, edges = np.histogram(r_num, bins=nbins, range=(rmin, rmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers_kpc = centers * u.kpc
    edges_kpc = edges * u.kpc
    pdf_per_kpc = pdf / u.kpc
    return centers_kpc, pdf_per_kpc, edges_kpc


def sample_rmean_batch(
    M,
    z,
    c,
    n=1000,
    seed=None,
    mass_def="200c",
    nr=4096,
    rmax_factor_Rvir=5.0,
    return_pdf=True,
    nbins=30,
):
    r"""Convenience wrapper: sample orbits and compute the PDF of \langle r \rangle.

    This is primarily used for quick demonstrations. It draws (E, L, Rc) using
    :func:`sample_EL_from_eta_Rc`, computes time-averaged radii, and (optionally)
    returns a histogram-based PDF.

    Returns
    -------
    rmean : ndarray
        Time-averaged radii.
    extras : dict
        Dictionary containing the sampled quantities and (optionally) ``r_pdf``.
    """
    E, L, eta, Rc, halo = sample_EL_from_eta_Rc(M, z, c, n=n, seed=seed, mass_def=mass_def)
    rmean, rp, ra, Tr = time_averaged_radius_batch_halo(E, L, halo, nr=nr, rmax_factor_Rvir=rmax_factor_Rvir)
    extras = dict(E=E, L=L, eta=eta, Rc=Rc, rp=rp, ra=ra, Tr=Tr, halo=halo)
    if return_pdf:
        centers_kpc, pdf_per_kpc, edges_kpc = radial_pdf(rmean, nbins=nbins)
        extras["r_pdf"] = dict(centers=centers_kpc, pdf=pdf_per_kpc, edges=edges_kpc)
    return rmean, extras


def plot_r_pdf(extras, *, xunit="kpc", logy=False, ax=None):
    """Plot a radial PDF produced by :func:`radial_pdf`.

    Parameters
    ----------
    extras : dict
        Must contain ``extras['r_pdf']`` with keys ``centers`` and ``pdf``.
    xunit : {'kpc', 'Mpc'}
        Unit to display on the x-axis.
    logy : bool
        If True, use a log y-axis.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axis; otherwise create a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for plotting.
    """
    u = units_and_constants()
    conv = dict(kpc=u.kpc, Mpc=u.Mpc)[xunit]
    centers = extras["r_pdf"]["centers"]
    pdf = extras["r_pdf"]["pdf"]
    x = np.asarray(centers, float) / float(conv)
    y = np.asarray(pdf, float) * float(conv)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, drawstyle="steps-mid")
    ax.set_xlabel(f"r [{xunit}]")
    ax.set_ylabel(f"PDF [1/{xunit}]")
    if logy:
        ax.set_yscale("log")
    return ax


# =============================================================================
# Orbit phase-table cache (for fast multi-time evaluation)
# =============================================================================

def _g_vr2_cache(r: np.ndarray | float, E: float, L: float, rs: float, Vs2: float) -> np.ndarray | float:
    r"""Squared radial speed helper: ``v_r^2(r)``.

    Computes
    $$v_r^2 = 2\,(E-\Phi(r)) - L^2/r^2,$$
    where ``Phi_NFW`` is used for the potential.

    Notes
    -----
    This is kept separate from other helpers to reduce the chance of accidental
    behavior changes in the cache machinery.
    """
    # identical to _g_vr2, but kept separate to avoid accidental edits
    phi = Phi_NFW(r, rs, Vs2)
    return 2.0 * (E - phi) - (L * L) / (r * r)


def turning_points_NFW_cache(
    E: float,
    L: float,
    r0: float,
    halo: dict,
    *,
    rmin: float | None = None,
    rmax_limit: float | None = None,
    expand: float = 1.6,
    max_iter: int = 64,
) -> tuple[float, float]:
    """Find pericenter and apocenter for a (possibly) bound NFW orbit.

    Parameters
    ----------
    E, L : float
        Specific energy and angular momentum.
    r0 : float
        Reference radius that must lie in the allowed region (``v_r^2(r0) >= 0``).
    halo : dict
        Halo dictionary containing ``rs``, ``Vs2``, and ``Rvir``.
    rmin : float or None
        Lower search bound; defaults to a small fraction of (Rvir, rs).
    rmax_limit : float or None
        Upper search bound used for bracketing the apocenter.
    expand : float
        Multiplicative factor used when expanding the apocenter bracket.
    max_iter : int
        Max number of bracket expansion steps.

    Returns
    -------
    (rp, ra) : tuple of float
        Pericenter and apocenter. Returns (nan, nan) if bracketing fails.
    """
    rs = halo["rs"]
    Vs2 = halo["Vs2"]
    Rvir = halo["Rvir"]

    E = float(E)
    L = float(L)
    r0 = float(r0)

    if rmin is None:
        rmin = max(1e-12 * float(Rvir), 1e-10 * float(rs))
    if rmax_limit is None:
        rmax_limit = 50.0 * float(Rvir)

    g0 = _g_vr2_cache(r0, E, L, rs, Vs2)
    if not np.isfinite(g0) or g0 < 0.0:
        return np.nan, np.nan

    tol = 1e-14 * max(1.0, abs(2.0 * E))
    if abs(g0) < tol:
        rin = max(rmin, r0 * 0.9)
        rout = min(rmax_limit, r0 * 1.1)
        gin = _g_vr2_cache(rin, E, L, rs, Vs2)
        gout = _g_vr2_cache(rout, E, L, rs, Vs2)
        if (np.isfinite(gin) and gin <= 0.0) and (np.isfinite(gout) and gout <= 0.0):
            return r0, r0

    gL = _g_vr2_cache(rmin, E, L, rs, Vs2)
    if np.isfinite(gL) and (gL < 0.0) and (g0 > 0.0):
        rp = cast(float, brentq(lambda rr: _g_vr2_cache(rr, E, L, rs, Vs2), rmin, r0, maxiter=200))
    else:
        rp = float(rmin)

    r_hi = r0 * 1.05
    for _ in range(max_iter):
        if r_hi >= rmax_limit:
            return np.nan, np.nan
        g_hi = _g_vr2_cache(r_hi, E, L, rs, Vs2)
        if np.isfinite(g_hi) and (g_hi < 0.0):
            break
        r_hi *= expand

    r_lo = r0 if abs(g0) >= tol else (r0 * 1.000001)
    ra = cast(float, brentq(lambda rr: _g_vr2_cache(rr, E, L, rs, Vs2), r_lo, r_hi, maxiter=200))
    return rp, ra


def build_orbit_cache(E, L, r0, halo, *, n_theta=512, rmin=None, rmax_limit=None):
    """Precompute per-orbit phase tables for fast evaluation of r(t).

    For each orbit i, this builds a monotonic grid r(θ) on the outward leg using
    the mapping ``r = rp + (ra-rp) sin^2 θ``. It then tabulates
    ``tau(θ)=∫ (dr/dθ)/v_r dθ`` and stores:

    - turning points ``rp, ra``
    - radial period ``Tr = 2 * tau_end``
    - the initial phase ``tau0_out`` corresponding to the provided ``r0``.

    The resulting cache allows :func:`evolve_r_from_cache` to compute r(t) for
    many times without redoing root finds / quadratures.

    Parameters
    ----------
    E, L, r0 : array-like
        Arrays of equal length defining each orbit.
    halo : dict
        Halo dictionary.
    n_theta : int
        Resolution of the θ grid per orbit.
    rmin, rmax_limit : float or None
        Bounds forwarded to :func:`turning_points_NFW_cache`.

    Returns
    -------
    dict
        Cache dictionary containing arrays and per-orbit lookup tables.
    """
    rs = halo["rs"]
    Vs2 = halo["Vs2"]

    E = np.asarray(E, float)
    L = np.asarray(L, float)
    r0 = np.asarray(r0, float)
    assert E.shape == L.shape == r0.shape
    n = E.size

    theta = np.linspace(0.0, 0.5 * np.pi, int(n_theta), dtype=float)
    dth = theta[1] - theta[0]
    sin2 = np.sin(theta) ** 2
    sin2th = np.sin(2.0 * theta)

    rp = np.full(n, np.nan, float)
    ra = np.full(n, np.nan, float)
    Tr = np.full(n, np.nan, float)
    tau0_out = np.full(n, np.nan, float)
    tau_end = np.full(n, np.nan, float)

    r_grid = np.full((n, n_theta), np.nan, float)
    tau_grid = np.full((n, n_theta), np.nan, float)

    for i in trange(n, desc="Building orbit cache"):
        _rp, _ra = turning_points_NFW_cache(E[i], L[i], r0[i], halo, rmin=rmin, rmax_limit=rmax_limit)
        if not (np.isfinite(_rp) and np.isfinite(_ra)):
            continue

        rp[i], ra[i] = _rp, _ra

        if abs(_ra - _rp) < 1e-15 * max(1.0, abs(r0[i])):
            Tr[i] = 0.0
            tau0_out[i] = 0.0
            tau_end[i] = 0.0
            r_grid[i, :] = r0[i]
            tau_grid[i, :] = 0.0
            continue

        rg = _rp + (_ra - _rp) * sin2
        drd = (_ra - _rp) * sin2th

        g = _g_vr2_cache(rg, E[i], L[i], rs, Vs2)
        g = np.maximum(g, 0.0)
        vr = np.sqrt(g)

        integrand = np.zeros_like(theta)
        good = vr > 0.0
        integrand[good] = drd[good] / vr[good]

        dtau = 0.5 * (integrand[:-1] + integrand[1:]) * dth
        tg = np.empty_like(theta)
        tg[0] = 0.0
        tg[1:] = np.cumsum(dtau)

        te = tg[-1]
        T = 2.0 * te
        if not (np.isfinite(T) and T > 0.0):
            continue

        t0 = np.interp(r0[i], rg, tg)

        r_grid[i, :] = rg
        tau_grid[i, :] = tg
        tau_end[i] = te
        Tr[i] = T
        tau0_out[i] = t0

    return dict(
        E=E,
        L=L,
        r0=r0,
        rp=rp,
        ra=ra,
        Tr=Tr,
        r_grid=r_grid,
        tau_grid=tau_grid,
        tau_end=tau_end,
        tau0_out=tau0_out,
        halo=halo,
        theta=theta,
    )


def evolve_r_from_cache(cache, t, *, sign_mode="in", seed=None, signs=None):
    """Evaluate radii r(t) for all cached orbits.

    Parameters
    ----------
    cache : dict
        Output of :func:`build_orbit_cache`.
    t : float
        Time since the initial condition, in the internal time unit.
    sign_mode : {'in', 'out', 'random', 'both'}
        Sets the initial radial direction at ``t=0``:
        ``'out'`` uses the outward-leg phase, ``'in'`` uses the inward-leg phase,
        ``'random'`` chooses per particle, and ``'both'`` stacks out+in samples.
    seed : int or None
        RNG seed used when ``sign_mode='random'``.
    signs : array-like or None
        Optional explicit +/-1 array (length N) overriding RNG in random mode.

    Returns
    -------
    ndarray
        Radii at time ``t`` for each orbit (or 2N values for ``sign_mode='both'``).
        Invalid orbits are returned as NaN.
    """
    rng = np.random.default_rng(seed)

    Tr = cache["Tr"]
    te = cache["tau_end"]
    t0 = cache["tau0_out"]
    rg = cache["r_grid"]
    tg = cache["tau_grid"]

    n = Tr.size
    t = float(t)

    valid = np.isfinite(Tr) & (Tr > 0.0) & np.isfinite(te) & np.isfinite(t0)
    circ = np.isfinite(Tr) & (Tr == 0.0) & np.isfinite(cache["r0"])

    def _one_branch(tau0):
        rt = np.full(n, np.nan, float)
        rt[circ] = cache["r0"][circ]

        idx = np.where(valid)[0]
        if idx.size == 0:
            return rt

        Tr_i = Tr[idx]
        te_i = te[idx]
        tau_t = (tau0[idx] + t) % Tr_i

        for k, i in enumerate(idx):
            if tau_t[k] <= te_i[k]:
                rt[i] = np.interp(tau_t[k], tg[i], rg[i])
            else:
                rt[i] = np.interp(Tr_i[k] - tau_t[k], tg[i], rg[i])
        return rt

    if sign_mode == "out":
        return _one_branch(t0)
    if sign_mode == "in":
        return _one_branch(Tr - t0)
    if sign_mode == "random":
        if signs is None:
            s = rng.choice(np.array([-1.0, 1.0]), size=n)
        else:
            s = np.asarray(signs, float)
            assert s.shape == (n,)
        tau0 = np.where(s >= 0.0, t0, Tr - t0)
        return _one_branch(tau0)
    if sign_mode == "both":
        out = _one_branch(t0)
        inn = _one_branch(Tr - t0)
        return np.concatenate([out, inn])

    raise ValueError("sign_mode must be one of: in, out, random, both")


def t_dyn_from_cache(cache, *, method="median_Tr"):
    """Define a representative dynamical time from cached radial periods.

    Parameters
    ----------
    cache : dict
        Output of :func:`build_orbit_cache`.
    method : {'median_Tr', 'mean_Tr'}
        How to summarize the distribution of radial periods.

    Returns
    -------
    float
        Dynamical time in the internal unit system (time unit attached by u.s).
    """
    u = units_and_constants()
    Tr = cache["Tr"]
    Tr = Tr[np.isfinite(Tr) & (Tr > 0.0)]
    if Tr.size == 0:
        raise RuntimeError("No valid bound orbits to define t_dyn.")
    if method == "median_Tr":
        return np.median(Tr) * u.s
    if method == "mean_Tr":
        return np.mean(Tr) * u.s
    raise ValueError("method must be 'median_Tr' or 'mean_Tr'")


# =============================================================================
# Main
# =============================================================================

def main():
    """Entry point for the standalone demo.

    This performs the end-to-end workflow described in the module docstring:
    sample orbits, build phase-table cache, define ``t_dyn``, and plot the
    evolution of the radial PDF as a function of ``tau=t/t_dyn``.

    Notes
    -----
    By default this runs a fairly heavy computation and calls ``plt.show()``.
    For quick automated checks (headless environments), set the environment
    variable ``SASHIMI_FAST=1``. This will reduce the problem size and save the
    figure to ``log/radial_pdf_evolution_fast.png`` instead of showing it.
    """
    from tqdm import tqdm

    fast = os.environ.get("SASHIMI_FAST", "0") not in {"0", "false", "False", ""}
    if fast:
        plt.switch_backend("Agg")

    u = units_and_constants()
    Gyr = 1e9 * u.yr

    M = 1e12 * u.Msun
    z = 0.5
    c = 8.0

    n_orbit = 4096 if fast else 60000
    n_theta = 512 if fast else 4096
    n_bins = 50
    n_tau = 6 if fast else 10

    # 1) sample once
    #    - (E,L) are sampled stochastically from the same (Rc,eta) recipe
    #    - all particles are placed at r0=Rvir(z) at t=0 (inward crossing via sign_mode='in')
    print("Sampling orbits (stochastic E,L; placed at r0=Rvir)...")
    E, L, eta, Rc, r0, halo = sample_EL_from_eta_Rc_placed_at_Rvir(
        M, z, c, n=n_orbit, seed=1, mass_def="200c"
    )

    # 2) cache once
    print("Building orbit cache...")
    cache = build_orbit_cache(E, L, r0, halo, n_theta=n_theta)

    # 3) dynamical time
    t_dyn = t_dyn_from_cache(cache, method="median_Tr")
    print("t_dyn =", float(t_dyn / Gyr), "Gyr")

    # 4) sweep tau
    tau_list = np.linspace(0.0, 2.0, n_tau)
    t_list = [tau * t_dyn for tau in tau_list]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")

    for tau, t in tqdm(zip(tau_list, t_list), total=len(tau_list), desc="Evolving radial PDF"):
        r_t = evolve_r_from_cache(cache, t, sign_mode="in")
        r_t = r_t[np.isfinite(r_t)]

        centers_kpc, pdf_per_kpc, edges_kpc = radial_pdf(r_t, nbins=n_bins)
        extras = {"r_pdf": {"centers": centers_kpc, "pdf": pdf_per_kpc, "edges": edges_kpc}}
        ax = plot_r_pdf(extras, xunit="kpc", logy=False, ax=ax)

    # color only the last len(tau_list) lines added
    lines = ax.get_lines()[-len(tau_list) :]
    for tau, line in tqdm(zip(tau_list, lines), total=len(tau_list), desc="Coloring lines"):
        line.set_color(cmap(tau / max(tau_list)))

    # Avoid tau=0 (delta-like) PDF dominating the y-axis scaling
    if len(lines) >= 2:
        y_max = 0.0
        for line in lines[1:]:
            yd = np.asarray(line.get_ydata(), dtype=float)
            if yd.size:
                y_max = max(y_max, float(np.nanmax(yd)))
        if np.isfinite(y_max) and y_max > 0.0:
            ax.set_ylim(0.0, 1.05 * y_max)

    ax.legend([f"τ = {tau:.2f}" for tau in tau_list])
    ax.set_title("Radial PDF evolution (time in units of t_dyn = median Tr)")

    if fast:
        os.makedirs("log", exist_ok=True)
        outpath = os.path.join("log", "radial_pdf_evolution_fast.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", outpath)
    else:
        plt.show()


if __name__ == "__main__":
    main()
