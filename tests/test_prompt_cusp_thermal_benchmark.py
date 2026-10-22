"""Check the decoupling epoch that fixes the primordial cusp phase-space bound."""

import numpy as np
from scipy.constants import Boltzmann, electron_volt
from scipy.integrate import quad

from sashimi_c.prompt_cusps import prompt_cusps


def _entropy_momentum_integrand(momentum, mass_ratio, statistics):
    r"""Evaluate the ideal-gas entropy contribution at a dimensionless momentum.

    Parameters
    ----------
    momentum : float
        Nonnegative momentum divided by temperature, ``p/T``, with c=k_B=1.
    mass_ratio : float
        Positive rest energy divided by temperature, ``m/T``, in matching units.
    statistics : {1, -1}
        Fermi-Dirac (+1) or Bose-Einstein (-1) occupation at zero chemical
        potential.

    Returns
    -------
    float
        Dimensionless integrand. Integrating from zero to infinite momentum
        and multiplying by ``45*g/(4*pi**4)`` gives this species' entropy
        degrees of freedom, where g includes antiparticles and spin.

    Notes
    -----
    The independent reference uses ``s=(rho+P)/T`` and the momentum-space
    energy and pressure integrals in Husdal [1]_. Exponentially decaying
    occupations avoid overflow at high momentum. The helper does not model
    chemical potentials, plasma interactions or out-of-equilibrium heating.
    The calling test sets quadrature tolerances; these do not bound the
    physical error of the ideal-gas approximation.

    References
    ----------
    .. [1] Husdal (2016), On Effective Degrees of Freedom in the Early
       Universe, Eqs. (6)-(8) and (11)-(13).
       https://arxiv.org/abs/1609.04979
    """
    energy = np.hypot(momentum, mass_ratio)
    boltzmann_weight = np.exp(-energy)
    occupation = boltzmann_weight / (1.0 + statistics * boltzmann_weight)
    return momentum**2 * (energy + momentum**2 / (3.0 * energy)) * occupation


def test_prompt_cusp_decoupling_scale_factor_obeys_entropy_conservation():
    r"""Check the stored cusp decoupling epoch against thermal entropy conservation.

    Notes
    -----
    Physical question: is the scale factor used in the primordial maximum
    phase-space density consistent with the adopted 30 MeV kinetic-decoupling
    temperature [1]_? The real ``prompt_cusps`` constructor supplies T_kd and
    a_kd; no spectrum, peak sample or frozen catalogue is used as a reference.

    Independently integrate Fermi-Dirac/Bose-Einstein entropy for electrons,
    muons and pions, with photons and three neutrino flavours treated as
    massless. The masses in MeV (0.511, 105.7, 135.0, 139.6) are the rounded
    values in Husdal [2]_, Table 2; degeneracies include antiparticles. This
    gives g_star_s(T_kd) about 12.36. Adopt T_gamma0=2.7255 K, rounded from
    Fixsen [3]_, and g_star_s0=43/11 from instantaneous neutrino decoupling
    [2]_, Eq. (39). With a_today=1, Eq. (28) gives
    ``a_kd=(T_gamma0/T_kd)*(g_star_s0/g_star_s_kd)**(1/3)``, using the same
    energy units for both temperatures.

    Pass conditions: the calculated a_kd agrees with the published
    5.33e-12 [4]_ within its half-last-digit rounding interval (5e-15), and
    the actual stored a_kd agrees within its two-significant-digit interval
    (5e-14). Either mismatch fails. These absolute bounds concern published
    and stored rounding, not quadrature error or cosmological accuracy.
    Negative controls omitting entropy dilution or the muon/pion tails must
    lie outside the stored-value interval. The adaptive quadrature target is
    1e-10 in each dimensionless integral, much smaller than these intervals.

    This tests a fixed benchmark under adiabatic expansion, zero chemical
    potentials and ideal-gas statistics. It does not test non-instantaneous
    neutrino decoupling, finite-temperature interaction corrections, a
    different dark sector, or the calibration of cusp luminosities.

    References
    ----------
    .. [1] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark
       matter halos: Implications for annihilation boost, Sections 2.1-2.2,
       especially Eq. (2.5).
       https://arxiv.org/abs/2601.19863
    .. [2] Husdal (2016), On Effective Degrees of Freedom in the Early
       Universe, Tables 1-2, Eqs. (11)-(13), (28) and (39).
       https://arxiv.org/abs/1609.04979
    .. [3] Fixsen (2009), The Temperature of the Cosmic Microwave Background.
       https://arxiv.org/abs/0911.1955
    .. [4] Stucker et al. (2023), The effect of stellar encounters on the
       dark matter annihilation signal from prompt cusps, Section 2.1, p. 3.
       https://arxiv.org/abs/2301.04670
    """
    cusp = prompt_cusps(k_fs=1.06e6)
    temperature_mev = cusp.T_kd / cusp.GeV * 1e3
    species = (
        (0.511, 4, 1),
        (105.7, 4, 1),
        (135.0, 1, -1),
        (139.6, 2, -1),
    )
    entropy_dof = 2.0 + 6.0 * 7.0 / 8.0
    for mass_mev, degeneracy, statistics in species:
        integral, _ = quad(
            _entropy_momentum_integrand,
            0.0,
            np.inf,
            args=(mass_mev / temperature_mev, statistics),
            epsabs=1e-10,
            epsrel=1e-10,
        )
        entropy_dof += 45.0 * degeneracy / (4.0 * np.pi**4) * integral

    photon_temperature_mev = 2.7255 * Boltzmann / electron_volt / 1e6
    temperature_ratio = photon_temperature_mev / temperature_mev
    entropy_dof_today = 43.0 / 11.0
    expected_a_kd = temperature_ratio * np.cbrt(entropy_dof_today / entropy_dof)

    np.testing.assert_allclose(expected_a_kd, 5.33e-12, rtol=0.0, atol=5e-15)
    np.testing.assert_allclose(cusp.a_kd, expected_a_kd, rtol=0.0, atol=5e-14)
    without_muons_or_pions = temperature_ratio * np.cbrt(entropy_dof_today / 10.75)
    assert abs(cusp.a_kd - temperature_ratio) > 5e-14
    assert abs(cusp.a_kd - without_muons_or_pions) > 5e-14
