"""
Prompt-cusp extension for SASHIMI-C.

This module contains everything specific to *prompt cusps* (relevant only for the
dark-matter annihilation signal). It is imported lazily by ``sashimi_c`` and is
NOT needed for standard SASHIMI-C usage:

  * ``build_ps_interpolators`` is called by ``halo_model.__init__`` only when
    ``prompt_cusps=True`` -- it replaces the Ludlow sigma(M) fit with sigma(M)
    computed from the CAMB linear matter power spectrum (free-streaming cut off).
  * ``prompt_cusps`` is the class used by
    ``subhalo_observables.annihilation_boost_factor_prompt_cusps``.

If ``prompt_cusps=False`` (the default) this module is never imported, so the
non-annihilation pipeline is completely unaffected.

Reference: Ando et al., arXiv:2601.19863.
"""
import numpy as np
import os
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.special import erf

from sashimi_c import cosmology
from ._data import data_directory, require_input


def build_ps_interpolators(cosmo, k_fs, filter='Sharp-k', alpha=1.8,
                           datafile=None):
    r"""Build the cutoff-spectrum variance and mass-derivative interpolators for prompt-cusp halo abundances.

    Parameters
    ----------
    cosmo : object
        Background and unit provider with OmegaM, rhocrit0, Mpc, h, Msun and data_dir.
    k_fs : float
        Free-streaming cutoff in inverse internal length.
    filter : {'Sharp-k', 'TopHat'}
        Variance window; sharp-k is the default.
    alpha : float
        Dimensionless sharp-k cutoff multiplier in kmax=alpha/R; default 1.8.
    datafile : path-like or None
        CAMB table with k in h/Mpc and P in (Mpc/h)**3; None selects the bundled table.

    Returns
    -------
    sigma_interp, dsdm_interp : scipy.interpolate.interp1d
        Cubic interpolators versus log10(M/Msun), returning dimensionless sigma and d(sigma**2)/dM in inverse internal mass.

    Notes
    -----
    The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
    alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
    simulation calibration.

    The tabulated mass range is 1e-12 to 1e17 solar masses. The free-streaming factor multiplies power as exp[-(k/k_fs)**2]. Grid sizes are numerical choices; there is no accuracy guarantee outside the input-table coverage. Missing tables raise FileNotFoundError.

    References
    ----------

    .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
        https://arxiv.org/abs/2601.19863
    """
    source = require_input(datafile if datafile is not None else data_directory(cosmo.data_dir)/"Planck2018_CAMB_extrap.dat")
    k, Pk = np.loadtxt(source, unpack=True)
    k  = k * (cosmo.Mpc / cosmo.h) ** -1
    Pk = Pk * (cosmo.Mpc / cosmo.h) ** 3
    Pk = Pk * np.exp(-(k / k_fs) ** 2)

    def sigma_calc(M):
        r"""Evaluate the selected cutoff-power variance window on the physical halo mass grid.

        Parameters
        ----------
        M : numpy.ndarray
            Positive physical mass in the internal units supplied by cosmo; the top-hat derivative expects an array.

        Returns
        -------
        values : numpy.ndarray
            Dimensionless rms density fluctuation, with the shape of M.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        Uses the enclosing power table and alpha/R sharp-k cutoff or spherical top-hat. The derivative is of sigma squared, not of sigma.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863
        """
        R = np.cbrt(3. * M / (4. * np.pi * cosmo.OmegaM * cosmo.rhocrit0))
        if filter == 'TopHat':
            R         = R[..., np.newaxis]
            W         = 3 * (np.sin(k * R) / (k * R) - np.cos(k * R)) / (k * R) ** 2
            integrand = k ** 3 * W ** 2 * Pk / (2. * np.pi ** 2)
            sigma_sq  = simpson(integrand, x=np.log(k), axis=-1)
        elif filter == 'Sharp-k':
            kmin      = k[0]
            kmax      = alpha / R
            kk        = np.logspace(np.log10(kmin), np.log10(kmax), 1000)
            fint      = interp1d(np.log(k), np.log(Pk))
            Pkk       = np.exp(fint(np.log(kk)))
            integrand = kk ** 3 * Pkk / (2. * np.pi ** 2)
            sigma_sq  = simpson(integrand, x=np.log(kk), axis=0)
        return np.sqrt(sigma_sq)

    def dsdm_calc(M):
        r"""Differentiate the selected cutoff-power variance with respect to halo mass.

        Parameters
        ----------
        M : numpy.ndarray
            Positive physical mass in the internal units supplied by cosmo; the top-hat derivative expects an array.

        Returns
        -------
        values : numpy.ndarray
            Derivative of variance in inverse internal mass, with the mass-grid shape.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        Uses the enclosing power table and alpha/R sharp-k cutoff or spherical top-hat. The derivative is of sigma squared, not of sigma.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863
        """
        if filter == 'TopHat':
            M         = M.reshape(-1, 1)
            R         = np.cbrt(3. * M / (4. * np.pi * cosmo.OmegaM * cosmo.rhocrit0))
            x         = k * R
            dxdM      = x / (3. * M)
            W         = 3 * (np.sin(x) / x - np.cos(x)) / x ** 2
            dWdx      = (9. * np.cos(x)) / (x ** 3) - (9. * np.sin(x)) / (x ** 4) + (3. * np.sin(x)) / (x ** 2)
            integrand = k ** 3 * Pk / (2. * np.pi ** 2) * 2. * W * dWdx * dxdM
            return simpson(integrand, x=np.log(k), axis=-1)
        elif filter == 'Sharp-k':
            R     = np.cbrt(3. * M / (4. * np.pi * cosmo.OmegaM * cosmo.rhocrit0))
            kmax  = alpha / R
            fint  = interp1d(np.log(k), np.log(Pk))
            Pkmax = np.exp(fint(np.log(kmax)))
            return -kmax ** 3 * Pkmax / M / (6. * np.pi ** 2)

    M_dummy = np.logspace(-12, 17, 2000) * cosmo.Msun
    sigma_interp = interp1d(np.log10(M_dummy / cosmo.Msun), sigma_calc(M_dummy),
                            kind='cubic', bounds_error=False, fill_value=None)
    dsdm_interp  = interp1d(np.log10(M_dummy / cosmo.Msun), dsdm_calc(M_dummy),
                            kind='cubic', bounds_error=False, fill_value=None)
    return sigma_interp, dsdm_interp


class prompt_cusps(cosmology):
    
    
    r"""Model the Gaussian peaks that seed prompt cusps and their annihilation luminosity.

    Parameters
    ----------
    k_fs : float
        Comoving free-streaming cutoff in inverse internal length.
    data_dir : path-like or None
        Directory containing the linear power spectrum and optional peak cache.

    Attributes
    ----------
    m_chi, T_kd : float
        Particle mass and kinetic-decoupling temperature in the internal mass/energy convention.
    a_kd : float
        Adopted kinetic-decoupling scale factor.
    k_fs : float
        Comoving cutoff in inverse internal length.

    Notes
    -----
    The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
    alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
    simulation calibration.

    The thermal benchmark and cusp/core prescription follow Delos & White; the luminosity
    constant 0.531 follows Stucker et al. and Ando et al., Eq. (2.8). The retained
    diagnostic fit with coefficient 0.08 is from Delos & White; it is evaluated internally
    but is not returned by cusp_properties.

    The state fixes a 100 GeV particle, 30 MeV kinetic-decoupling benchmark and a_kd=5.3e-12. The latter is a retained upstream benchmark value; its exact numerical provenance has not been independently recovered from the cited paper. Use powerspectrum and sigma_function before sampling or evaluating cusp_properties. This is the small-scale cusp prescription, not a new calibration.

    References
    ----------

    .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
        https://arxiv.org/abs/2601.19863

    .. [cite-delos] Delos & White (2023), Prompt cusps and the dark matter annihilation signal.
        https://arxiv.org/abs/2209.11237

    .. [cite-stucker] Stucker et al. (2023), The effect of stellar encounters on the dark matter annihilation signal from prompt cusps.
        https://arxiv.org/abs/2301.04670
    """
    def __init__(self, k_fs, *, data_dir=None):
        r"""Set the thermal benchmark and data location for the prompt-cusp calculation.

        Parameters
        ----------
        k_fs : float
            Comoving cutoff in inverse internal length.
        data_dir : path-like or None
            Root for the power table and peak caches.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        The thermal benchmark and cusp/core prescription follow Delos & White; the luminosity
        constant 0.531 follows Stucker et al. and Ando et al., Eq. (2.8). The retained
        diagnostic fit with coefficient 0.08 is from Delos & White; it is evaluated internally
        but is not returned by cusp_properties.

        The benchmark fixes m_chi=100 GeV, T_kd=0.03 GeV and a_kd=5.3e-12. The precise source of a_kd has not been independently recovered; it is preserved as an upstream input, not newly attributed to a paper.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863

        .. [cite-delos] Delos & White (2023), Prompt cusps and the dark matter annihilation signal.
            https://arxiv.org/abs/2209.11237

        .. [cite-stucker] Stucker et al. (2023), The effect of stellar encounters on the dark matter annihilation signal from prompt cusps.
            https://arxiv.org/abs/2301.04670
        """
        cosmology.__init__(self, data_dir=data_dir)
        
        self.GeV      = self.Msun/1.118e57
        self.m_chi    = 100.*self.GeV
        self.T_kd     = 0.03*self.GeV
        self.a_kd     = 5.3e-12
        self.k_fs     = k_fs
        
        
    def powerspectrum(self, k, z=0.):
        r"""Rescale the z=31 dimensionless spectrum to the requested epoch and apply free streaming.

        Parameters
        ----------
        k : array_like
            Comoving wavenumbers in inverse internal length; k*self.Mpc is in Mpc**(-1).
        z : float
            Dimensionless observation redshift; the small-scale growth approximation is assumed.

        Returns
        -------
        Delta2 : numpy.ndarray
            Dimensionless k**3*P(k)/(2*pi**2), broadcast over k.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        Uses small-scale growth Delta2 proportional to a**(2*g), g=0.901, and the Gaussian power cutoff. The input spectrum is logarithmically interpolated and extrapolated; this is not a full Boltzmann evolution at arbitrary redshift.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863
        """
        _z         = 31.
        _a         = 1./(1.+_z)
        a          = 1./(1.+z)
        g          = 0.901
        _k,_Delta2 = np.loadtxt(require_input(self.data_dir/'powerspectrum31.txt'),unpack=True,delimiter=',')
        _k        *= self.Mpc**-1
        _Delta2   *= (a/_a)**(2.*g)
        fint       = interp1d(np.log(_k),np.log(_Delta2),fill_value='extrapolate')
        Delta2     = np.exp(fint(np.log(k)))
        Delta2     = Delta2*np.exp(-(k/self.k_fs)**2)
        
        return Delta2

    
    def sigma_function(self, k=np.logspace(-1.,7.,num=1000)):
        #Spectral parameters
        r"""Integrate the three spectral moments needed for BBKS peak statistics.

        Parameters
        ----------
        k : array_like
            One-dimensional increasing wavenumbers as numerical values in Mpc**(-1); converted to internal inverse length.

        Returns
        -------
        sigma : numpy.ndarray, shape (3,)
            Moments sigma_j=sqrt(integral Delta2*k**(2*j) dlnk); units internal length**(-j).
        R_s : float
            sqrt(3)*sigma_1/sigma_2, a comoving internal length.
        gamma : float
            sigma_1**2/(sigma_0*sigma_2), dimensionless.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
        statistics; finite grids and Monte Carlo sizes are numerical choices.

        The default finite grid runs from 0.1 to 1e7 Mpc**(-1). It is a numerical integration domain, not a fitted physical cutoff.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863

        .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
            https://doi.org/10.1086/164143
        """
        k     = k/self.Mpc
        sigma = np.empty(3)
        for j in np.arange(3):
            y        = self.powerspectrum(k=k)*k**(2*j)
            sigma_sq = simpson(y,x=np.log(k))
            sigma[j] = np.sqrt(sigma_sq)

        #Comoving characteristic correlation length
        R_s      = np.sqrt(3.)*sigma[1]/sigma[2]

        #Deviation from pure power-law behaviour
        gamma    = sigma[1]**2/(sigma[0]*sigma[2])

        return np.array(sigma), R_s, gamma
    

    def cusps_Monte_Carlo(self, length=10000, nu_max=10.0, x_max=10.0, N_nu=1001, N_x=1000,
                          rng_seed=None, verbose=False,
    ):
        r"""Sample peak height, curvature and tidal shape for the ellipsoidal prompt-cusp collapse model.

        Parameters
        ----------
        length : int
            Number of requested independent Monte Carlo draws.
        nu_max, x_max : float
            Upper dimensionless height and curvature limits.
        N_nu, N_x : int
            Numbers of grid points for height and curvature.
        rng_seed : int or None
            Seed for a local NumPy random generator.
        verbose : bool
            Print the number of collapse-root attempts.

        Returns
        -------
        x_random, nu_random, e_random, p_random, fec_random : numpy.ndarray, shape (length,)
            Dimensionless curvature, height, ellipticity, prolateness and collapse factor; failed/unsupported collapse factors are NaN.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
        statistics; finite grids and Monte Carlo sizes are numerical choices.

        The collapse coefficients 0.47 and 0.615 and conditional ellipticity/prolateness law
        come from Sheth, Mo & Tormen, Eq. (3) and Appendix A. The e**2-p*abs(p)<0.26 cut is the
        prompt-cusp adoption described by Ando et al.

        The peak grid is normalized over finite positive nu and x. Shape draws use u=p/e and e**2 drawn from a Gamma distribution with shape 5/2. Peaks outside e**2-p*abs(p)<0.26, or failed positive collapse roots, retain NaN collapse factors. Invalid probability normalization raises ValueError; successful sampling does not establish cusp-abundance calibration.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863

        .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
            https://doi.org/10.1086/164143

        .. [cite-ellipsoid] Sheth, Mo & Tormen (2001), Ellipsoidal collapse and an improved model for the number and spatial distribution of dark matter haloes, Eq. (3), Appendix A.
            https://arxiv.org/abs/astro-ph/9907024
        """

        rng = np.random.default_rng(rng_seed)

        # --- spectral parameters ---
        sigma, R_s, gamma = self.sigma_function()

        # --- BBKS f(x) ---
        def f_x(x):
            r"""Evaluate the BBKS curvature weighting of maxima in a Gaussian density field.

            Parameters
            ----------
            x : array_like
                Dimensionless positive normalized peak curvature.

            Returns
            -------
            weight : array_like
                Dimensionless BBKS f(x), broadcast over x.

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
            statistics; finite grids and Monte Carlo sizes are numerical choices.

            The polynomial, error-function and exponential coefficients are those of BBKS Appendix A.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
                https://doi.org/10.1086/164143
            """
            return ((x**3 - 3.0 * x) / 2.0) * (
                erf(np.sqrt(5.0 / 2.0) * x) + erf(np.sqrt(5.0 / 8.0) * x)
            ) + np.sqrt(2.0 / (5.0 * np.pi)) * (
                (31.0 / 4.0 * x**2 + 8.0 / 5.0) * np.exp(-(5.0 / 8.0) * x**2)
                + (x**2 / 2.0 - 8.0 / 5.0) * np.exp(-(5.0 / 2.0) * x**2)
            )

        # --- BBKS d^2n/(dnu dx) (matches your ddn() structure) ---
        def d2n_dnudx(nu, x):
            r"""Evaluate the joint Gaussian peak density per height and curvature.

            Parameters
            ----------
            nu : array_like
                Dimensionless density height delta/sigma_0.
            x : array_like
                Dimensionless negative Laplacian divided by sigma_2; must broadcast with nu.

            Returns
            -------
            density : array_like
                d**2 n/(dnu dx) in inverse internal comoving volume.

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
            statistics; finite grids and Monte Carlo sizes are numerical choices.

            The enclosing spectral moments set R_s and gamma; the Gaussian conditional curvature law and BBKS f(x) determine the peak density.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
                https://doi.org/10.1086/164143
            """
            return (
                np.exp(-0.5 * nu**2)
                / ((2.0 * np.pi) ** 2 * R_s**3)
                * f_x(x)
                * np.exp(-0.5 * (x - gamma * nu) ** 2 / (1.0 - gamma**2))
                / np.sqrt(2.0 * np.pi * (1.0 - gamma**2))
            )

        # --- sample (nu, x) on a grid via CDF inversion (stable; avoids p-sum issues) ---
        nu_grid = np.linspace(1e-12, float(nu_max), int(N_nu))
        x_grid = np.linspace(1e-12, float(x_max), int(N_x))

        # weights on mesh: shape (N_nu, N_x)
        w = d2n_dnudx(nu_grid.reshape(-1, 1), x_grid)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)

        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0.0:
            raise ValueError("cusps_Monte_Carlo: peak weight integral is zero/invalid. Check sigma_function() output.")

        w_flat = w.ravel()
        cdf = np.cumsum(w_flat, dtype=np.float64)
        cdf /= cdf[-1]  # normalize exactly

        u01 = rng.random(int(length))
        idx = np.searchsorted(cdf, u01, side="right")
        idx = np.clip(idx, 0, w_flat.size - 1)

        i_nu = idx // x_grid.size
        i_x = idx % x_grid.size

        nu_random = nu_grid[i_nu].astype(float)
        x_random = x_grid[i_x].astype(float)

        # -------------------------------------------------------------------------
        # --- DW-consistent sampling of (e, p) given nu via (u=p/e, t=e^2) ----------
        # Target: f(u|nu) ∝ (1-u^2) (3+u^2)^(-5/2) on u∈[-1,1]
        # Two-stage rejection:
        #   stage1: Uniform[-1,1] accepted with (1-u^2)  -> proposal ∝ (1-u^2)
        #   stage2: accept with (3/(3+u^2))^(5/2)       -> adds (3+u^2)^(-5/2)
        def draw_u(n):
            r"""Draw the shape ratio p/e for the adopted ellipsoidal-collapse distribution.

            Parameters
            ----------
            n : int
                Number of accepted shape-ratio draws.

            Returns
            -------
            u : numpy.ndarray, shape (n,)
                Dimensionless ratios in [-1,1].

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            The collapse coefficients 0.47 and 0.615 and conditional ellipticity/prolateness law
            come from Sheth, Mo & Tormen, Eq. (3) and Appendix A. The e**2-p*abs(p)<0.26 cut is the
            prompt-cusp adoption described by Ando et al.

            The two rejection steps sample density proportional to (1-u**2)*(3+u**2)**(-5/2), obtained by integrating the adopted conditional ellipticity law. Oversampling sizes only affect execution.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-ellipsoid] Sheth, Mo & Tormen (2001), Ellipsoidal collapse and an improved model for the number and spatial distribution of dark matter haloes, Eq. (3), Appendix A.
                https://arxiv.org/abs/astro-ph/9907024
            """
            out = np.empty(n, dtype=float)
            filled = 0
            while filled < n:
                m = (n - filled) * 2 + 64
                uu = rng.uniform(-1.0, 1.0, size=m)

                # stage 1: accept with (1-u^2)
                a1 = rng.random(m) < (1.0 - uu**2)
                uu = uu[a1]
                if uu.size == 0:
                    continue

                # stage 2: accept with (3/(3+u^2))^(5/2)
                # This is always <= 1 since 3/(3+u^2) ∈ [3/4, 1]
                a2 = rng.random(uu.size) < (3.0 / (3.0 + uu**2)) ** 2.5
                uu = uu[a2]
                if uu.size == 0:
                    continue

                take = min(uu.size, n - filled)
                out[filled : filled + take] = uu[:take]
                filled += take
            return out

        u_random = draw_u(int(length))

        # Given u and nu:
        # f(e|u,nu) ∝ e^4 exp(-alpha e^2), alpha=(5/2) nu^2 (3+u^2)
        # Let t=e^2 => f(t|u,nu) ∝ t^(3/2) exp(-alpha t)  => Gamma(k=5/2, theta=1/alpha)
        alpha = 2.5 * (nu_random**2) * (3.0 + u_random**2)
        if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
            raise ValueError("cusps_Monte_Carlo: alpha invalid (non-finite or non-positive).")

        t = rng.gamma(shape=2.5, scale=1.0 / alpha, size=int(length))
        e_random = np.sqrt(t)
        p_random = e_random * u_random

        # -------------------------------------------------------------------------
        # --- collapse factor f_ec (keep your existing definition/condition) -------
        def fec_fun(fec, e, p):
            r"""Form the ellipsoidal-collapse root equation for a single peak shape.

            Parameters
            ----------
            fec : array_like
                Trial spherical-to-ellipsoidal collapse correction.
            e, p : float
                Dimensionless ellipticity and prolateness.

            Returns
            -------
            residual : array_like
                fec-1-0.47*[5*(e**2-p*abs(p))*fec**2]**0.615; the desired root is zero.

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            The collapse coefficients 0.47 and 0.615 and conditional ellipticity/prolateness law
            come from Sheth, Mo & Tormen, Eq. (3) and Appendix A. The e**2-p*abs(p)<0.26 cut is the
            prompt-cusp adoption described by Ando et al.

            A numerical root is sought only after applying the retained prompt-cusp shape cut.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-ellipsoid] Sheth, Mo & Tormen (2001), Ellipsoidal collapse and an improved model for the number and spatial distribution of dark matter haloes, Eq. (3), Appendix A.
                https://arxiv.org/abs/astro-ph/9907024
            """
            return fec - 1.0 - 0.47 * (5.0 * (e**2 - p * np.abs(p)) * fec**2) ** 0.615

        fec_random = np.full(int(length), np.nan, dtype=float)

        # Condition used in your code: e^2 - p|p| < 0.26
        cond = (e_random**2 - p_random * np.abs(p_random)) < 0.26
        idx_solve = np.where(cond)[0]
        if verbose:
            print(f"cusps_Monte_Carlo: solving f_ec for {idx_solve.size}/{length} samples")

        for j in idx_solve:
            ej = float(e_random[j])
            pj = float(p_random[j])
            try:
                sol = root(lambda f: fec_fun(f, ej, pj), x0=1.0)
                if sol.success and np.isfinite(sol.x[0]) and sol.x[0] > 0.0:
                    fec_random[j] = float(sol.x[0])
                else:
                    fec_random[j] = np.nan
            except Exception:
                fec_random[j] = np.nan

        return x_random, nu_random, e_random, p_random, fec_random
        
        
    def _load_or_generate_prompt_cusp_mc(
        self,
        length=10000,
        mc_cache_file=None,
        regenerate=False,
        rng_seed=None,
        verbose=False,
    ):
        r"""Load cached Monte Carlo samples for prompt-cusp peak parameters, or generate and cache them.

        The cached file contains 5 columns in the following order:
            x, nu, e, p, f_ec

        Parameters
        ----------
        length : int
            Number of Monte Carlo samples to generate if cache is missing or regeneration is requested.
        mc_cache_file : str
            Path to the cache file.
        regenerate : bool
            If True, ignore any existing cache and regenerate samples.
        rng_seed : int or None
            RNG seed passed to the generator for reproducibility.
        verbose : bool
            If True, print basic cache/generation info.

        Returns
        -------
        x_random, nu_random, e_random, p_random, fec_random : 1D arrays

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863
        """

        cache_path = str(self.data_dir/"prompt_cusps"/"prompt_cusps_Monte_Carlo.txt") if mc_cache_file is None else str(mc_cache_file)
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if (not regenerate) and os.path.exists(cache_path):
            try:
                data = np.loadtxt(cache_path)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] != 5:
                    raise ValueError(f"Expected 5 columns (x,nu,e,p,f_ec) in {cache_path}, got {data.shape[1]}")
                x_random, nu_random, e_random, p_random, fec_random = data.T
                if verbose:
                    print(f"prompt_cusps: loaded Monte Carlo cache: {cache_path} (N={x_random.size})")
                return x_random, nu_random, e_random, p_random, fec_random
            except Exception as exc:
                if verbose:
                    print(f"prompt_cusps: failed to load cache {cache_path} ({exc}); regenerating.")

        # generate and cache
        x_random, nu_random, e_random, p_random, fec_random = self.cusps_Monte_Carlo(
            length=int(length),
            rng_seed=rng_seed,
            verbose=verbose,
        )

        header = (
            "x, nu, e, p, f_ec\n"
            "Generated by prompt_cusps.cusps_Monte_Carlo and cached for reproducibility." 
        )
        out = np.column_stack([x_random, nu_random, e_random, p_random, fec_random])
        np.savetxt(cache_path, out, header=header)
        if verbose:
            print(f"prompt_cusps: saved Monte Carlo cache: {cache_path} (N={out.shape[0]})")

        return x_random, nu_random, e_random, p_random, fec_random

    def cusp_properties(
        self,
        f_surv=1.,
        z=0.,
        mc_length=10000,
        mc_cache_file=None,
        regenerate_mc=False,
        rng_seed=None,
        verbose=False,
    ):

        r"""Compute the collapsed peak fraction and individual prompt-cusp annihilation integrals.

        Parameters
        ----------
        f_surv : float
            Dimensionless survival factor used in internal population luminosity diagnostics.
        z : float
            Dimensionless observation redshift; the small-scale growth approximation is assumed.
        mc_length : int
            Number of generated peaks if a cache is not reused.
        mc_cache_file : path-like or None
            Five-column peak cache; None selects the data-directory default.
        regenerate_mc : bool
            Regenerate rather than reuse the peak cache.
        rng_seed : int or None
            Seed used only when generating the Monte Carlo sample.
        verbose : bool
            Print generation/cache status.

        Returns
        -------
        f_coll : float
            Fraction of sampled peaks passing the epoch and shape collapse conditions.
        J_cusp : numpy.ndarray
            Integral of rho**2 over physical volume for each collapsed cusp, in internal mass**2/length**3.

        Notes
        -----
        The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
        alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
        simulation calibration.

        Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
        statistics; finite grids and Monte Carlo sizes are numerical choices.

        The thermal benchmark and cusp/core prescription follow Delos & White; the luminosity
        constant 0.531 follows Stucker et al. and Ando et al., Eq. (2.8). The retained
        diagnostic fit with coefficient 0.08 is from Delos & White; it is evaluated internally
        but is not returned by cusp_properties.

        The cusp normalization is A=24*rho_dm0*a_coll**(-3/2)*R**(3/2), the outer radius is 0.11*a_coll*R, and the phase-space core uses the coefficient 3e-5. The luminosity uses 4*pi*A**2*[0.531+ln(rcusp/rcore)]. Internal population-average and 0.08 fit diagnostics are not return values. The historical implementation assumes a nonempty collapsed sample; its empty-sample path is not repaired by this documentation change.

        References
        ----------

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863

        .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
            https://doi.org/10.1086/164143

        .. [cite-delos] Delos & White (2023), Prompt cusps and the dark matter annihilation signal.
            https://arxiv.org/abs/2209.11237

        .. [cite-stucker] Stucker et al. (2023), The effect of stellar encounters on the dark matter annihilation signal from prompt cusps.
            https://arxiv.org/abs/2301.04670
        """
        x_random, nu_random, e_random, p_random, fec_random = self._load_or_generate_prompt_cusp_mc(
            length=mc_length,
            mc_cache_file=mc_cache_file,
            regenerate=regenerate_mc,
            rng_seed=rng_seed,
            verbose=verbose,
        )

        sigma,R_s,gamma         = self.sigma_function()
        sigma_0,sigma_1,sigma_2 = sigma[0],sigma[1],sigma[2]

        #num same as the one used in the sampling  
        def f_x(x):
            r"""Evaluate the BBKS curvature weighting of maxima in a Gaussian density field.

            Parameters
            ----------
            x : array_like
                Dimensionless positive normalized peak curvature.

            Returns
            -------
            weight : array_like
                Dimensionless BBKS f(x), broadcast over x.

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
            statistics; finite grids and Monte Carlo sizes are numerical choices.

            The polynomial, error-function and exponential coefficients are those of BBKS Appendix A.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
                https://doi.org/10.1086/164143
            """
            return (x**3-3.*x)/2.*(erf(np.sqrt(5./2.)*x)+erf(np.sqrt(5./8.)*x)) \
                    +np.sqrt(2./(5.*np.pi))*((31./4.*x**2+8./5.)*np.exp(-(5./8.*x**2)) \
                    +(x**2/2.-8./5.)*np.exp(-(5./2.*x**2)))

        def ddn(nu,x):
            r"""Evaluate the joint Gaussian peak density per height and curvature.

            Parameters
            ----------
            nu : array_like
                Dimensionless density height delta/sigma_0.
            x : array_like
                Dimensionless negative Laplacian divided by sigma_2; must broadcast with nu.

            Returns
            -------
            density : array_like
                d**2 n/(dnu dx) in inverse internal comoving volume.

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
            statistics; finite grids and Monte Carlo sizes are numerical choices.

            The enclosing spectral moments set R_s and gamma; the Gaussian conditional curvature law and BBKS f(x) determine the peak density.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
                https://doi.org/10.1086/164143
            """
            return np.exp(-nu**2/2.)/((2.*np.pi)**2*R_s**3)*f_x(x) \
                    *np.exp(-0.5*(x-gamma*nu)**2/(1.-gamma**2)) \
                    /np.sqrt(2.*np.pi*(1.-gamma**2))
        
        #Differential number density of peaks
        def n_peaks(nu=np.linspace(1.e-10,10.,num=1000),x=np.linspace(1.e-10,10.,num=1001)):
            r"""Integrate the BBKS peak density over the selected positive height/curvature domain.

            Parameters
            ----------
            nu, x : numpy.ndarray
                One-dimensional increasing dimensionless height and curvature grids.

            Returns
            -------
            number_density : float
                Number of peaks per internal comoving volume over these finite bounds.

            Notes
            -----
            The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
            alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
            simulation calibration.

            Peak-height/curvature weighting and spectral moments use the BBKS Gaussian-peak
            statistics; finite grids and Monte Carlo sizes are numerical choices.

            Simpson integration uses the joint BBKS density. Changing bounds changes the included peak population; grid refinement alone cannot correct omitted tails.

            References
            ----------

            .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
                https://arxiv.org/abs/2601.19863

            .. [cite-bbks] Bardeen, Bond, Kaiser & Szalay (1986), The Statistics of Peaks of Gaussian Random Fields, Appendix A.
                https://doi.org/10.1086/164143
            """
            dndnudx_ = ddn(nu,x.reshape(-1,1))
            n_peaks  = simpson(simpson(dndnudx_,nu,axis=-1),x)
            return n_peaks
        
        #Scale factor
        a  = 1./(1.+z)

        #Growth index at small scales
        g  = 0.901

        #Critical density contrast for a spherical collapse
        delta_ec = 1.686

        #Averaged peak's characteristics
        #delta           = nu_average*sigma_0
        #dd_delta        = x_average*sigma_2
        #delta_random    = nu_random*sigma_0             #Used for one of the conditions
        
        delta      = nu_random*sigma_0
        dd_delta   = -x_random*sigma_2

        collapsed  = (delta*a**g>1.686*fec_random)*(e_random**2-p_random*np.abs(p_random)<0.26)
        f_coll     = np.sum(collapsed)/len(collapsed)
        N_cusps_M  = f_surv*f_coll*n_peaks()/(self.OmegaC*self.rhocrit0)
        N_peaks_M  = n_peaks()/(self.OmegaC*self.rhocrit0)
        #print(N_peaks_M/(1.e3*self.Msun**-1))
        
        collapsed_indeces = np.where(collapsed)
        
        delta      = delta[collapsed_indeces]
        dd_delta   = dd_delta[collapsed_indeces]
        fec_random = fec_random[collapsed_indeces]
        #x_random, nu_random, e_random, p_random, fec_random = x_random[collapsed_indeces], nu_random[collapsed_indeces], e_random[collapsed_indeces], p_random[collapsed_indeces], fec_random[collapsed_indeces]
        #Cusp properties
        R       = np.sqrt(np.abs(delta/dd_delta))
        delta_a = delta*a**g
        a_coll  = (fec_random*delta_ec/delta_a)**(1./g)*a
        Aec     = 24.*(self.OmegaC*self.rhocrit0)*a_coll**-1.5*R**1.5
        rcusp   = 0.11*a_coll*R
        f_max   = (2.*np.pi)**(-3./2.)*(self.m_chi/self.T_kd)**(3./2.)*self.OmegaC*self.rhocrit0*self.a_kd**-3
        rcore   = (3.e-5*self.G**-3*f_max**-2*Aec**-1*self.c**6)**(1./4.5)
        Mcusp   = 8.*np.pi/3.*Aec*rcusp**1.5
        #print(np.mean(R/self.pc), np.mean(rcore/self.pc), np.mean(rcusp/self.pc), np.mean(rcusp/rcore))

        #Calculate number of cusps

        idx = np.where(collapsed)[0]
        if idx.size==0:
            J_cusp = 0.
        else:
            J_cusp       = 4*np.pi*Aec**2*(0.531+np.log(rcusp/rcore))
            J_cusp_mean  = np.mean(J_cusp)

        #Calculate the annihilation factor per unit dark matter mass
        J_cusps_M = J_cusp_mean*N_cusps_M
        J_cusps_fit = 0.08*f_surv*(np.log(self.m_chi*self.T_kd/self.GeV**2)+36.)**5*(self.OmegaC*self.rhocrit0)
        
        #print(np.mean(R)/self.pc,np.mean(a_coll),f_coll)
        #print(np.mean(Aec)/(self.Msun*self.pc**-1.5),np.mean(rcore)/self.pc,np.mean(rcusp)/self.pc,np.mean(Mcusp)/self.Msun)
        #print(f_coll, np.mean(J_cusp)/(self.Msun**2*self.pc**-3))
        #print(np.mean(Mcusp)/self.Msun,np.mean(a_coll),np.mean(R)/self.pc)
        #print(f_coll,N_cusps_M/(1.e3*self.Msun**-1),J_cusps_M/(self.Msun*self.pc**-3),J_cusps_fit/(self.Msun*self.pc**-3))

        return f_coll, J_cusp
