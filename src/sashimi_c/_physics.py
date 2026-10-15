"""CDM-owned calibrated physics, tidal solvers and observable formulae.

The population controller and numerical primitives belong to ITAMAE. This
module contains no old catalog implementation or reproduction-mode switches.
All historical formula bodies are extracted from the reviewed C model; units
are initialized against the chosen, compatible cosmology before any work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from itamae.cosmology import NativeFlatLCDM
from itamae.evolution import solve_evolution
from itamae.protocols import CosmologyBackend
from itamae.types import WeightedSubhaloCatalog
from scipy import optimize
from scipy.integrate import cumulative_trapezoid

from ._boost import interpolate_boost_tables, validate_order
from ._data import data_directory
from .picard_tidal_stripping import (
    PicardTidalStrippingTable,
    get_picard_solver,
    physics_key,
)

_CALIBRATED_OMEGA_M = 0.315
_CALIBRATED_H = 0.674


class CDMUnits:
    r"""Define internal dimensional scales used by the retained CDM halo equations.

    Notes
    -----
    These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
    base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
    observable boundaries. The fitted relations and mass definitions remain model choices,
    not consequences of the shared numerical executor.
    """

    def __init__(self):
        r"""Initialize Msun, Mpc and second base scales and their conversion factors.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        self.Mpc = 1.0
        self.kpc = self.Mpc / 1000.0
        self.pc = self.kpc / 1000.0
        self.cm = self.pc / 3.086e18
        self.km = 1.0e5 * self.cm
        self.s = 1.0
        self.yr = 3.15576e7 * self.s
        self.Msun = 1.0
        self.gram = self.Msun / 1.988e33
        self.c = 2.9979e10 * self.cm / self.s
        self.G = 4.30091e-9 * self.Mpc * (self.km / self.s) ** 2 / self.Msun


class CDMPhysics(CDMUnits):
    r"""Literature host history, concentration, EPS and structure relations.

    Parameters
    ----------
    prompt_cusps : bool
        Enable the variant prompt-cusp prescription and its explicitly supplied spectrum
        inputs.
    k_fs_Mpc : float
        Free-streaming wavenumber numerical value in Mpc^-1 for the prompt-cusp spectrum
        prescription.
    filter : str
        Named physical smoothing/filter prescription used in the prompt-cusp calculation.
    alpha : float
        Dimensionless shape parameter of the selected prompt-cusp filter/transfer
        prescription.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.
    data_dir : pathlib.Path or str, optional
        Explicit root of the required scientific input tables or generated products; no
        substitute spectrum is selected implicitly.

    Notes
    -----
    The adopted background Omega_m0=0.315 and h=0.674 (and Omega_b0=0.049 where used) is a
    rounded Planck-2018 parameter choice. It is not evidence that every retained halo fit
    was calibrated in this background.

    If prompt cusps are enabled, the benchmark cutoff k_fs=1.06e6 Mpc**(-1) and sharp-k
    multiplier alpha=1.8 follow the prompt-cusp model of Ando et al.; these are separate
    from the CDM variance fit used by default.

    These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
    base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
    observable boundaries. The fitted relations and mass definitions remain model choices,
    not consequences of the shared numerical executor.

    References
    ----------

    .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
        https://arxiv.org/abs/1807.06209

    .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3 and 3.1.
        https://arxiv.org/abs/2601.19863
    """

    def __init__(
        self,
        prompt_cusps=False,
        k_fs_Mpc=1.06e6,
        filter="Sharp-k",
        alpha=1.8,
        *,
        cosmology_backend=None,
        data_dir=None,
    ):
        r"""Bind the calibrated CDM halo relations to a compatible background and optional cusp
        spectrum.

        Parameters
        ----------
        prompt_cusps : bool
            Enable the variant prompt-cusp prescription and its explicitly supplied spectrum
            inputs.
        k_fs_Mpc : float
            Free-streaming wavenumber numerical value in Mpc^-1 for the prompt-cusp spectrum
            prescription.
        filter : str
            Named physical smoothing/filter prescription used in the prompt-cusp calculation.
        alpha : float
            Dimensionless shape parameter of the selected prompt-cusp filter/transfer
            prescription.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.
        data_dir : pathlib.Path or str, optional
            Explicit root of the required scientific input tables or generated products; no
            substitute spectrum is selected implicitly.

        Notes
        -----
        The adopted background Omega_m0=0.315 and h=0.674 (and Omega_b0=0.049 where used) is a
        rounded Planck-2018 parameter choice. It is not evidence that every retained halo fit
        was calibrated in this background.

        If prompt cusps are enabled, the benchmark cutoff k_fs=1.06e6 Mpc**(-1) and sharp-k
        multiplier alpha=1.8 follow the prompt-cusp model of Ando et al.; these are separate
        from the CDM variance fit used by default.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
            https://arxiv.org/abs/1807.06209

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3 and 3.1.
            https://arxiv.org/abs/2601.19863
        """
        backend = cosmology_backend or NativeFlatLCDM(
            omega_m0=_CALIBRATED_OMEGA_M, h=_CALIBRATED_H
        )
        self._validate_migration_cosmology(backend)
        self.itamae_cosmology = backend
        CDMUnits.__init__(self)
        self.OmegaB = 0.049
        self.OmegaM = _CALIBRATED_OMEGA_M
        self.OmegaC = self.OmegaM - self.OmegaB
        self.OmegaL = 1.0 - self.OmegaM
        self.h = _CALIBRATED_H
        self._rho_crit_scale = 1.0
        self._synchronize_physics_constants()
        self.data_dir = data_directory(data_dir)
        self.prompt_cusps = prompt_cusps
        self.k_fs = k_fs_Mpc * self.Mpc**-1
        self.filter = filter
        self.alpha = alpha
        if prompt_cusps:
            from .prompt_cusps import build_ps_interpolators

            self.sigma_interp, self.dsdm_interp = build_ps_interpolators(
                self, self.k_fs, filter=filter, alpha=alpha
            )

    def _synchronize_physics_constants(self) -> None:
        r"""Align SASHIMI-C constants with the selected backend.

        Notes
        -----
        The adopted background Omega_m0=0.315 and h=0.674 (and Omega_b0=0.049 where used) is a
        rounded Planck-2018 parameter choice. It is not evidence that every retained halo fit
        was calibrated in this background.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
            https://arxiv.org/abs/1807.06209
        """
        backend = self.itamae_cosmology
        self.H0 = float(np.asarray(backend.H(0.0))) * self.km / self.s / self.Mpc
        backend_rho0 = (
            float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        )
        self.G = 3.0 * self.H0**2 / (8.0 * np.pi * backend_rho0)
        self._rho_crit_scale = 1.0
        self.rhocrit0 = backend_rho0

    @staticmethod
    def _validate_migration_cosmology(backend: Any) -> None:
        r"""Require a complete backend matching the calibrated CDM physical model.

        Parameters
        ----------
        backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.

        Notes
        -----
        The adopted background Omega_m0=0.315 and h=0.674 (and Omega_b0=0.049 where used) is a
        rounded Planck-2018 parameter choice. It is not evidence that every retained halo fit
        was calibrated in this background.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
            https://arxiv.org/abs/1807.06209
        """
        if not isinstance(backend, CosmologyBackend):
            raise TypeError(
                "cosmology_backend must implement the ITAMAE cosmology protocol."
            )

        omega_m0 = float(np.asarray(backend.omega_m(0.0)))
        h = float(np.asarray(backend.H(0.0))) / 100.0
        if not np.isclose(omega_m0, _CALIBRATED_OMEGA_M, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "SASHIMI-C calibrated physics requires "
                f"OmegaM={_CALIBRATED_OMEGA_M}; received {omega_m0}."
            )
        if not np.isclose(h, _CALIBRATED_H, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "SASHIMI-C calibrated physics requires "
                f"h={_CALIBRATED_H}; received {h}."
            )

    def Hubble(self, z: Any) -> np.ndarray:
        r"""Return the Hubble rate in the internal inverse-second unit.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Expansion rate in inverse seconds, with the redshift shape.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        r"""Return critical density in the canonical solar masses per cubic megaparsec.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Backend critical density mapped to internal Msun/Mpc^3; the backend must actually
            supply canonical density.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        r"""Return the normalized linear growth factor.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Dimensionless linear growth normalized at z=0.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """
        return np.asarray(self.itamae_cosmology.growth_factor(z))

    def g(self, z):
        r"""Evaluate the flat matter-plus-Lambda squared expansion factor.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            OmegaM*(1+z)**3+OmegaL, dimensionless.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        return self.OmegaM * (1.0 + z) ** 3 + self.OmegaL

    def dDdz(self, z):
        r"""Differentiate the retained Carroll-Press-Turner growth approximation with redshift.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless derivative dD/dz, in the adopted CDM background.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """

        def dOdz(z):
            r"""Evaluate d Omega_Lambda(z)/dz for the analytic growth derivative.

            Parameters
            ----------
            z : float or array_like
                Dimensionless redshift. The cosmological background requires z > -1; population
                nodes use nonnegative redshifts.

            Returns
            -------
            array_like
                Redshift derivative of the dimensionless dark-energy fraction.

            Notes
            -----
            These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
            base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
            observable boundaries. The fitted relations and mass definitions remain model choices,
            not consequences of the shared numerical executor.
            """
            return (
                -self.OmegaL
                * 3
                * self.OmegaM
                * (1 + z) ** 2
                * (self.OmegaL + self.OmegaM * (1 + z) ** 3.0) ** -2
            )

        Omega_Lz = self.OmegaL / (self.OmegaL + self.OmegaM * (1.0 + z) ** 3)
        Omega_Mz = 1 - Omega_Lz
        phiz = (
            Omega_Mz ** (4.0 / 7.0)
            - Omega_Lz
            + (1 + Omega_Mz / 2.0) * (1 + Omega_Lz / 70.0)
        )
        phi0 = (
            self.OmegaM ** (4.0 / 7.0)
            - self.OmegaL
            + (1 + self.OmegaM / 2.0) * (1 + self.OmegaL / 70.0)
        )
        dphidz = dOdz(z) * (
            -4.0 / 7.0 * Omega_Mz ** (-3.0 / 7.0)
            + (Omega_Mz - Omega_Lz) / 140.0
            + 1.0 / 70.0
            - 3.0 / 2.0
        )
        return (phi0 / self.OmegaM) * (
            -dOdz(z) / (phiz * (1 + z))
            - Omega_Mz * (dphidz * (1 + z) + phiz) / phiz**2 / (1 + z) ** 2
        )

    def sigmaMz(self, M, z):
        r"""Evaluate the linear rms density fluctuation at halo mass and redshift.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            sigma(M,0)*D(z), using the Ludlow fit or the explicitly selected cusp spectrum.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        if not self.prompt_cusps:
            return self.sigma_Ludlow(M) * self.growthD(z)
        else:
            return self.sigma_interp(np.log10(M / self.Msun)) * self.growthD(z)

    def dsdm(self, M, z):
        r"""Evaluate the mass derivative of linear variance, including growth squared.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            dS(M,z)/dM in inverse internal mass units, with S=sigma**2.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        if not self.prompt_cusps:
            return self.dsdm_Ludlow(M) * self.growthD(z) ** 2
        else:
            return self.dsdm_interp(np.log10(M / self.Msun)) * self.growthD(z) ** 2

    def sigma_Ludlow(self, M):
        r"""Evaluate the retained Ludlow et al. (2016) CDM rms-fluctuation fit.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.

        Returns
        -------
        array_like
            Dimensionless sigma(M,0); the fitted mass variable uses 1e10 Msun/h.

        Notes
        -----
        The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
        fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
        Derivatives are analytic derivatives of this retained fit.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """

        def xi(M):
            r"""Form the dimensionless inverse mass used by the Ludlow variance fit.

            Parameters
            ----------
            M : float or array_like
                Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
                value in solar masses.

            Returns
            -------
            array_like
                xi=(1e10 Msun/h)/M.

            Notes
            -----
            The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
            fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
            Derivatives are analytic derivatives of this retained fit.

            These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
            base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
            observable boundaries. The fitted relations and mass definitions remain model choices,
            not consequences of the shared numerical executor.

            References
            ----------

            .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
                https://arxiv.org/abs/1601.02624
            """
            return (M / ((1.0e10 * self.Msun) / self.h)) ** -1

        return (
            22.26
            * xi(M) ** 0.292
            / (1.0 + 1.53 * xi(M) ** 0.275 + 3.36 * xi(M) ** 0.198)
        )

    def dsdm_Ludlow(self, M):
        r"""Differentiate the squared Ludlow rms-fluctuation fit with respect to mass.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.

        Returns
        -------
        array_like
            dS(M,0)/dM=2*sigma*d sigma/dM in inverse internal mass units.

        Notes
        -----
        The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
        fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
        Derivatives are analytic derivatives of this retained fit.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """

        def xi(M):
            r"""Form the dimensionless inverse mass used by the analytic variance derivative.

            Parameters
            ----------
            M : float or array_like
                Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
                value in solar masses.

            Returns
            -------
            array_like
                xi=(1e10 Msun/h)/M.

            Notes
            -----
            The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
            fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
            Derivatives are analytic derivatives of this retained fit.

            These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
            base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
            observable boundaries. The fitted relations and mass definitions remain model choices,
            not consequences of the shared numerical executor.

            References
            ----------

            .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
                https://arxiv.org/abs/1601.02624
            """
            return (M / ((1.0e10 * self.Msun) / self.h)) ** -1

        dsdsigma = 2.0 * self.sigma_Ludlow(M)
        dxidm = -1.0e10 * self.Msun / self.h / M**2
        dsigmadxi = self.sigma_Ludlow(M) * (
            0.292 / xi(M)
            - (0.275 * 1.53 * xi(M) ** -0.725 + 0.198 * 3.36 * xi(M) ** -0.802)
            / (1.0 + 1.53 * xi(M) ** 0.275 + 3.36 * xi(M) ** 0.198)
        )
        return dsdsigma * dsigmadxi * dxidm

    def deltac_func(self, z):
        r"""Evaluate the linearly extrapolated spherical-collapse barrier for EPS.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            delta_c(z)=1.686/D(z), dimensionless.

        Notes
        -----
        The adopted spherical-collapse barrier is delta_c(z)=1.686/D(z), as used in Ludlow et
        al., Appendix C. This approximation is separate from any mass-dependent WDM collapse
        correction.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        return 1.686 / self.growthD(z)

    def s_func(self, M):
        r"""Evaluate the present-day linear variance used throughout the EPS kernel.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.

        Returns
        -------
        array_like
            S(M)=sigma(M,0)**2, dimensionless.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        return self.sigmaMz(M, 0) ** 2

    def fc(self, x):
        r"""Evaluate the dimensionless NFW enclosed-mass normalization.

        Parameters
        ----------
        x : array_like
            Positive NFW radius ratio r/r_s or concentration used to normalize enclosed mass.

        Returns
        -------
        array_like
            f(x)=ln(1+x)-x/(1+x), with x a radius ratio or concentration.

        Notes
        -----
        The enclosed-mass factor ln(1+c)-c/(1+c) follows the NFW density profile. Algebraic
        inversion tolerances concern this profile identity, not agreement with simulated halos.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107
        """
        return np.log(1 + x) - x * pow(1 + x, -1)

    def Delc(self, x):
        r"""Evaluate the retained virial overdensity relative to critical density.

        Parameters
        ----------
        x : array_like
            Dimensionless Omega_m(z)-1, not a radius ratio.

        Returns
        -------
        array_like
            Delta_c=18*pi**2+82*x-39*x**2 with x=Omega_m(z)-1.

        Notes
        -----
        The flat-universe expression 18*pi**2+82*x-39*x**2, x=Omega_m(z)-1, is Bryan & Norman,
        Eq. (6); it defines overdensity relative to the critical density.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-virial] Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).
            https://arxiv.org/abs/astro-ph/9710107
        """
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    def conc200(self, M200, z):
        r"""Evaluate the retained Correa et al. (2015) median concentration relation.

        Parameters
        ----------
        M200 : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless c200, using the low-z branch through z=4 and the high-z branch above
            it.

        Notes
        -----
        The low/high-redshift concentration coefficients, with the branch at z=4, are the
        Planck-fit formulae in Correa et al., Appendix B1. Evaluation in the adopted
        Planck-2018-like background does not refit these earlier coefficients.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391
        """
        alpha_cMz_1 = 1.7543 - 0.2766 * (1.0 + z) + 0.02039 * (1.0 + z) ** 2
        beta_cMz_1 = 0.2753 + 0.00351 * (1.0 + z) - 0.3038 * (1.0 + z) ** 0.0269
        gamma_cMz_1 = -0.01537 + 0.02102 * (1.0 + z) ** -0.1475
        c_Mz_1 = np.power(
            10.0,
            alpha_cMz_1
            + beta_cMz_1
            * np.log10(M200 / self.Msun)
            * (1 + gamma_cMz_1 * np.log10(M200 / self.Msun) ** 2),
        )
        alpha_cMz_2 = 1.3081 - 0.1078 * (1.0 + z) + 0.00398 * (1.0 + z) ** 2
        beta_cMz_2 = 0.0223 - 0.0944 * (1.0 + z) ** -0.3907
        c_Mz_2 = pow(10, alpha_cMz_2 + beta_cMz_2 * np.log10(M200 / self.Msun))
        return np.where(z <= 4.0, c_Mz_1, c_Mz_2)

    def Mvir_from_M200(self, M200, z):
        r"""Convert M200 to virial mass by solving the same NFW enclosed-density equation.

        Parameters
        ----------
        M200 : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Virial mass in the input mass unit; fsolve uses the M200 radius as its initial
            guess.

        Notes
        -----
        The enclosed-mass factor ln(1+c)-c/(1+c) follows the NFW density profile. Algebraic
        inversion tolerances concern this profile identity, not agreement with simulated halos.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107
        """
        gz = self.g(z)
        c200 = self.conc200(M200, z)
        r200 = (3.0 * M200 / (4 * np.pi * 200 * self.rhocrit0 * gz)) ** (1.0 / 3.0)
        rs = r200 / c200
        fc200 = self.fc(c200)
        rhos = M200 / (4 * np.pi * rs**3 * fc200)
        Dc = self.Delc(self.OmegaM * (1.0 + z) ** 3 / self.g(z) - 1.0)
        rvir = optimize.fsolve(
            lambda r: (
                3.0 * (rs / r) ** 3 * self.fc(r / rs) * rhos - Dc * self.rhocrit0 * gz
            ),
            r200,
        )
        Mvir = 4 * np.pi * rs**3 * rhos * self.fc(rvir / rs)
        return Mvir

    def Mvir_from_M200_fit(self, M200, z):
        r"""Convert M200 to virial mass with the retained fitted NFW inversion.

        Parameters
        ----------
        M200 : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Virial mass under Delta_c(z), in the input mass unit.

        Notes
        -----
        The inverse mass-conversion approximation, including coefficients 0.5116, -0.4283,
        -3.13e-3 and -3.52e-5 where used, is Hu & Kravtsov, Appendix C. It is an approximate NFW
        inversion, not the exact root-solved conversion.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-conversion] Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.
            https://arxiv.org/abs/astro-ph/0203169
        """
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.OmegaM * (1.0 + z) ** 3 / self.g(z)

        def ffunc(x):
            r"""Express NFW mean enclosed density as a function of inverse concentration.

            Parameters
            ----------
            x : array_like
                Positive inverse NFW concentration r_s/r_delta.

            Returns
            -------
            array_like
                x**3*[ln(1+1/x)-1/(1+x)], with x the inverse concentration.

            Notes
            -----
            The inverse mass-conversion approximation, including coefficients 0.5116, -0.4283,
            -3.13e-3 and -3.52e-5 where used, is Hu & Kravtsov, Appendix C. It is an approximate NFW
            inversion, not the exact root-solved conversion.

            These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
            base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
            observable boundaries. The fitted relations and mass definitions remain model choices,
            not consequences of the shared numerical executor.

            References
            ----------

            .. [cite-conversion] Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.
                https://arxiv.org/abs/astro-ph/0203169
            """
            return np.power(x, 3.0) * (np.log(1.0 + 1.0 / x) - 1.0 / (1.0 + x))

        def xfunc(f):
            r"""Invert the dimensionless NFW mean-density mapping with the retained fit.

            Parameters
            ----------
            f : array_like
                Positive dimensionless mean-density mapping at the requested overdensity.

            Returns
            -------
            array_like
                Fitted inverse concentration corresponding to f.

            Notes
            -----
            The inverse mass-conversion approximation, including coefficients 0.5116, -0.4283,
            -3.13e-3 and -3.52e-5 where used, is Hu & Kravtsov, Appendix C. It is an approximate NFW
            inversion, not the exact root-solved conversion.

            These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
            base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
            observable boundaries. The fitted relations and mass definitions remain model choices,
            not consequences of the shared numerical executor.

            References
            ----------

            .. [cite-conversion] Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.
                https://arxiv.org/abs/astro-ph/0203169
            """
            p = a2 + a3 * np.log(f) + a4 * np.power(np.log(f), 2.0)
            return (
                np.power(a1 * np.power(f, 2.0 * p) + (3.0 / 4.0) ** 2, -0.5) + 2.0 * f
            )

        return (
            self.Delc(Oz - 1)
            / 200.0
            * M200
            * np.power(
                self.conc200(M200, z)
                * xfunc(self.Delc(Oz - 1) / 200.0 * ffunc(1.0 / self.conc200(M200, z))),
                -3.0,
            )
        )

    def Mzi(self, M0, z):
        r"""Evaluate the median M200 host history anchored by its z=0 mass.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            M200(z) in the same mass unit as M0.

        Notes
        -----
        The mass-history coefficients, including q = 4.137 / zf**0.9476 and the polynomial for
        zf, follow Correa et al., Section 3. Their original calibration is retained while the
        configured background is used; changing the background does not recalibrate the fit.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        a = 1.686 * np.sqrt(2.0 / np.pi) * self.dDdz(0) + 1.0
        zf = (
            -0.0064 * np.log10(M0 / self.Msun) ** 2
            + 0.0237 * np.log10(M0 / self.Msun)
            + 1.8837
        )
        q = 4.137 / zf**0.9476
        fM0 = (self.sigmaMz(M0 / q, 0) ** 2 - self.sigmaMz(M0, 0) ** 2) ** -0.5
        return M0 * np.power(1.0 + z, a * fM0) * np.exp(-fM0 * z)

    def Mzzi(self, M0, z, zi):
        r"""Evaluate the re-anchored host history from the reference epoch zi.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        zi : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            M200(z), anchored to Mzi(M0,zi), in the internal mass unit.

        Notes
        -----
        The mass-history coefficients, including q = 4.137 / zf**0.9476 and the polynomial for
        zf, follow Correa et al., Section 3. Their original calibration is retained while the
        configured background is used; changing the background does not recalibrate the fit.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        Mzi0 = self.Mzi(M0, zi)
        zf = (
            -0.0064 * np.log10(M0 / self.Msun) ** 2
            + 0.0237 * np.log10(M0 / self.Msun)
            + 1.8837
        )
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (
            1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1.0
        )
        beta = -fMzi
        return Mzi0 * np.power(1.0 + z - zi, alpha) * np.exp(beta * (z - zi))

    def dMdz(self, M0, z, zi):
        r"""Evaluate the retained virial host-growth factor used by the EPS abundance.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        zi : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            [beta+alpha/(1+z-zi)]*Mvir(z), in mass per unit redshift. The formula retains the
            original fitted virial-growth convention.

        Notes
        -----
        The mass-history coefficients, including q = 4.137 / zf**0.9476 and the polynomial for
        zf, follow Correa et al., Section 3. Their original calibration is retained while the
        configured background is used; changing the background does not recalibrate the fit.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        Mzi0 = self.Mzi(M0, zi)
        zf = (
            -0.0064 * np.log10(M0 / self.Msun) ** 2
            + 0.0237 * np.log10(M0 / self.Msun)
            + 1.8837
        )
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (
            1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1
        )
        beta = -fMzi
        Mzzidef = Mzi0 * (1.0 + z - zi) ** alpha * np.exp(beta * (z - zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef, z)
        return (beta + alpha / (1.0 + z - zi)) * Mzzivir


class CDMTidalKernels(CDMPhysics):
    r"""C-owned stripping laws, coefficient tables and solver choices.

    Parameters
    ----------
    M0 : float
        Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
        explicit epoch flags are described by the caller.
    z_min : float
        Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
    z_max : float
        Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
    n_z_interp : int
        Redshift-node count for the retained perturbative coefficient
        integrals/interpolants; independent of Picard endpoint-table settings.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.

    Notes
    -----
    These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
    base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
    observable boundaries. The fitted relations and mass definitions remain model choices,
    not consequences of the shared numerical executor.
    """

    def __init__(
        self, M0, z_min=0.0, z_max=7.0, n_z_interp=64, *, cosmology_backend=None
    ):
        r"""Initialize the C tidal law and its host-dependent perturbative coefficient tables.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z_min : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        z_max : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        n_z_interp : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        CDMPhysics.__init__(self, cosmology_backend=cosmology_backend)
        self.z_min = z_min
        self.z_max = z_max
        self.n_z_interp = n_z_interp
        self._picard_tables = {}
        self.M0 = M0

    @property
    def M0(self):
        r"""Return the z=0 host M200 anchoring the current tidal history.

        Returns
        -------
        float
            Reference host mass in the internal solar-mass unit.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        return self._M0

    @M0.setter
    def M0(self, value):
        """Set the reference host mass and rebuild its host-dependent tidal coefficient tables.

        Parameters
        ----------
        value : float
            New z=0 host M200 in internal solar-mass units.

        Notes
        -----
        Changing host mass clears retained Picard tables and recomputes the epsilon
        interpolants. Other mutable physical inputs must also be represented by the Picard
        physical-state key.
        """
        self._M0 = value
        if hasattr(self, "_picard_tables"):
            self._picard_tables.clear()
        self.reset_interpolation(
            z_max=self.z_max, z_min=self.z_min, n_z=self.n_z_interp
        )

    def reset_interpolation(self, z_max, z_min, n_z):
        r"""Rebuild the retained epsilon integrals and interpolation functions for the current host.

        This function is called when the mass of the host
        halo is changed.

        Parameters
        ----------
        z_max : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        z_min : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        _z, _eps_0 = self._eps_0(z_max, z_min, n_z)
        _, _eps_10, _eps_11 = self._eps_1(z_max, z_min, n_z)
        _, _eps_20, _eps_21, _eps_22 = self._eps_2(z_max, z_min, n_z)
        _, _eps_30, _eps_31, _eps_32, _eps_33 = self._eps_3(z_max, z_min, n_z)
        # get the interpolation functions as indefinite integrals
        self._eps_0_interp = lambda z: np.interp(z, _z[::-1], _eps_0[::-1])
        self._eps_10_interp = lambda z: np.interp(z, _z[::-1], _eps_10[::-1])
        self._eps_11_interp = lambda z: np.interp(z, _z[::-1], _eps_11[::-1])
        self._eps_20_interp = lambda z: np.interp(z, _z[::-1], _eps_20[::-1])
        self._eps_21_interp = lambda z: np.interp(z, _z[::-1], _eps_21[::-1])
        self._eps_22_interp = lambda z: np.interp(z, _z[::-1], _eps_22[::-1])
        self._eps_30_interp = lambda z: np.interp(z, _z[::-1], _eps_30[::-1])
        self._eps_31_interp = lambda z: np.interp(z, _z[::-1], _eps_31[::-1])
        self._eps_32_interp = lambda z: np.interp(z, _z[::-1], _eps_32[::-1])
        self._eps_33_interp = lambda z: np.interp(z, _z[::-1], _eps_33[::-1])
        # define the epsilon functions as definite integrals from za to z
        self.eps_0 = lambda _za, _z: self._eps_0_interp(_z) - self._eps_0_interp(_za)
        self.eps_10 = lambda _za, _z: self._eps_10_interp(_z) - self._eps_10_interp(_za)
        self.eps_11 = lambda _za, _z: self._eps_11_interp(_z) - self._eps_11_interp(_za)
        self.eps_20 = lambda _za, _z: self._eps_20_interp(_z) - self._eps_20_interp(_za)
        self.eps_21 = lambda _za, _z: self._eps_21_interp(_z) - self._eps_21_interp(_za)
        self.eps_22 = lambda _za, _z: self._eps_22_interp(_z) - self._eps_22_interp(_za)
        self.eps_30 = lambda _za, _z: self._eps_30_interp(_z) - self._eps_30_interp(_za)
        self.eps_31 = lambda _za, _z: self._eps_31_interp(_z) - self._eps_31_interp(_za)
        self.eps_32 = lambda _za, _z: self._eps_32_interp(_z) - self._eps_32_interp(_za)
        self.eps_33 = lambda _za, _z: self._eps_33_interp(_z) - self._eps_33_interp(_za)

    def Mzvir(self, z):
        r"""Evaluate the virial host history used in the subhalo-to-host mass ratio.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Virial host mass at z, in internal solar-mass units.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        Mz200 = self.Mzzi(self.M0, z, 0.0)
        Mvir = self.Mvir_from_M200_fit(Mz200, z)
        return Mvir

    def AMz(self, z):
        r"""Evaluate the fitted dimensionless amplitude of the C tidal mass-loss law.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            A(M_host,z) from the retained host-mass/redshift fit.

        Notes
        -----
        The adopted A(M,z) and zeta(M,z) coefficients are Hiroshima et al., Eqs. (23)-(24),
        calibrated for host masses 1e-6 to 1e16 solar masses and redshifts 0 to 7. Numerical
        integration or iteration controls do not extend that calibration or bound total model
        error.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-tidal] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Eqs. (23)-(24).
            https://arxiv.org/abs/1803.07691
        """
        log10a = (-0.0003 * np.log10(self.Mzvir(z) / self.Msun) + 0.02) * z + (
            0.011 * np.log10(self.Mzvir(z) / self.Msun) - 0.354
        )
        return 10.0**log10a

    def zetaMz(self, z):
        r"""Evaluate the fitted exponent of the tidal subhalo-to-host mass ratio.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless zeta(M_host,z).

        Notes
        -----
        The adopted A(M,z) and zeta(M,z) coefficients are Hiroshima et al., Eqs. (23)-(24),
        calibrated for host masses 1e-6 to 1e16 solar masses and redshifts 0 to 7. Numerical
        integration or iteration controls do not extend that calibration or bound total model
        error.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-tidal] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Eqs. (23)-(24).
            https://arxiv.org/abs/1803.07691
        """
        return (0.00012 * np.log10(self.Mzvir(z) / self.Msun) - 0.0033) * z + (
            -0.0011 * np.log10(self.Mzvir(z) / self.Msun) + 0.026
        )

    def tdynz(self, z):
        r"""Evaluate the virial dynamical time entering the fitted tidal loss law.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dynamical time in internal seconds, scaled by Delta_c(z) and H(z)/H0.

        Notes
        -----
        The normalization 1.628/h Gyr and factors (Delta_vir/178)**(-1/2)*(H/H0)**(-1) are Jiang
        & van den Bosch, Eq. (2), with Bryan--Norman virial overdensity.

        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.

        References
        ----------

        .. [cite-time] Jiang & van den Bosch (2016), Statistics of Dark Matter Substructure: I. Model and Universal Fitting Functions, Eq. (2).
            https://arxiv.org/abs/1403.6827

        .. [cite-virial] Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).
            https://arxiv.org/abs/astro-ph/9710107
        """
        Oz_z = self.OmegaM * (1.0 + z) ** 3 / self.g(z)
        return (
            1.628
            / self.h
            * (self.Delc(Oz_z - 1.0) / 178.0) ** -0.5
            / (self.Hubble(z) / self.H0)
            * 1.0e9
            * self.yr
        )

    def _get_picard_table(self, z_final):
        r"""Return a cached x3 Picard table for the requested final redshift.

        Parameters
        ----------
        z_final : float or array_like, optional
            Target redshift no greater than the accretion redshift. Explicit tables use one
            scalar target; None selects solver.z_min.

        Returns
        -------
        PicardTidalStrippingTable
            Explicit cached endpoint table valid for this physical state and final redshift.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        current_physics = physics_key(self)
        if getattr(self, "_picard_physics_key", None) != current_physics:
            self._picard_tables.clear()
            self._picard_physics_key = current_physics
        key = float(z_final)
        table = self._picard_tables.get(key)
        if table is None:
            table = PicardTidalStrippingTable(self, z_final=key)
            self._picard_tables[key] = table
        return table

    def subhalo_mass_stripped_picard_table(self, ma, za, z, **kwargs):
        r"""Calculate tidal mass loss with the precomputed Picard table.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        **kwargs : dict
            Controls for the selected Picard or ODE backend. Historical odeint Dfun callbacks
            use state-first signature; unsupported controller overrides are rejected.

        Returns
        -------
        float or numpy.ndarray
            Final bound masses in the ma unit; this endpoint interface does not return a
            redshift/mass tuple.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        return get_picard_solver(self).endpoint(ma, za, z, **kwargs)

    def msolve(self, m, z):
        r"""Evaluate the positive redshift derivative of the bound subhalo mass.

        Parameters
        ----------
        m : float or array_like
            Subhalo mass in the same internal unit as the host mass; masses must be positive for
            logarithmic tidal evolution.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            dm/dz=A*m*(m/M_host)**zeta/[t_dyn*H*(1+z)], in mass per unit redshift.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        return (
            self.AMz(z)
            * (m / self.tdynz(z))
            * (m / self.Mzvir(z)) ** self.zetaMz(z)
            / (self.Hubble(z) * (1 + z))
        )

    def subhalo_mass_stripped_odeint(self, ma, za, z0, **kwargs):
        r"""Integrate the owned tidal RHS through ITAMAE's odeint controller.

        Numerical options retain SciPy meanings. A supplied Dfun keeps this
        method's historical state-first signature; ITAMAE receives time first.
        The controller owns args/tfirst/full_output, which are not accepted as
        duplicate overrides. Repeated times and zero evolution are preserved.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z0 : float
            Target or reference redshift specified by the routine; the tidal ODE evolves from za
            down to this epoch.
        **kwargs : dict
            Controls for the selected Picard or ODE backend. Historical odeint Dfun callbacks
            use state-first signature; unsupported controller overrides are rejected.

        Returns
        -------
        float or numpy.ndarray
            Final bound masses in the ma unit; this endpoint interface does not return a
            redshift/mass tuple.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        options = dict(kwargs)
        rtol, atol = options.pop("rtol", None), options.pop("atol", None)
        jacobian = options.get("Dfun")
        if callable(jacobian):
            options["Dfun"] = lambda time, state: jacobian(state, time)
        zcalc = np.linspace(za, z0, 100)
        solution = solve_evolution(
            lambda time, state: self.msolve(state, time),
            ma,
            zcalc,
            method="odeint",
            rtol=rtol,
            atol=atol,
            odeint_options=options,
            allow_repeated_times=True,
        )
        return solution[-1]

    def Phi(self, z):
        """Evaluate the positive coefficient in d ln(m)/dz for the C tidal law.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            A/[t_dyn*H*(1+z)], dimensionless per unit redshift.

        Notes
        -----
        Phi includes dt/dz in magnitude. Evolving toward decreasing redshift therefore decreases
        mass; the physical cosmic-time law is dm/dt=-A*m*(m/M_host)**zeta/t_dyn.
        """
        return self.AMz(z) / self.tdynz(z) / self.Hubble(z) / (1 + z)

    def _eps_0(self, za, z, n_z=64):
        """Integrate the order-0 coefficients of the retained log-mass perturbation expansion.

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps0 : array
            epsilon0 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z = np.linspace(za, z, n_z)
        Phi_z = self.Phi(_z)
        return _z, cumulative_trapezoid(Phi_z, x=_z, initial=0)

    def _eps_1(self, za, z, n_z=64):
        """Integrate the order-1 coefficients of the retained log-mass perturbation expansion.

        The first order correction epsilon_1 is given by the following equation:
            epsilon_1 = epsilon_10 + epsilon_11 * ln_ma

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps10 : array
            epsilon10 array.
        eps11 : array
            epsilon11 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_10 = Phi_z * (eps_0 - ln_Mvir_z) * zeta_z
        integrand_11 = Phi_z * zeta_z
        integral_10 = cumulative_trapezoid(integrand_10, x=_z, initial=0)
        integral_11 = cumulative_trapezoid(integrand_11, x=_z, initial=0)
        return _z, integral_10, integral_11

    def _eps_2(self, za, z, n_z=64):
        """Integrate the order-2 coefficients of the retained log-mass perturbation expansion.

        The second order correction epsilon_2 is given by the following equation:
            epsilon_2 = epsilon_20 + epsilon_21 * ln_ma + epsilon_22 * ln_ma^2

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps20 : array
            epsilon20 array.
        eps21 : array
            epsilon21 array.
        eps22 : array
            epsilon22 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_20 = (
            Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) ** 2 / 2 + Phi_z * zeta_z * eps_10
        )
        integrand_21 = Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) + Phi_z * zeta_z * eps_11
        integrand_22 = Phi_z * zeta_z**2 / 2
        integral_20 = cumulative_trapezoid(integrand_20, x=_z, initial=0)
        integral_21 = cumulative_trapezoid(integrand_21, x=_z, initial=0)
        integral_22 = cumulative_trapezoid(integrand_22, x=_z, initial=0)
        return _z, integral_20, integral_21, integral_22

    def _eps_3(self, za, z, n_z=64):
        """Integrate the order-3 coefficients of the retained log-mass perturbation expansion.

        The third order correction epsilon_3 is given by the following equation:
            epsilon_3 = epsilon_30 + epsilon_31 * ln_ma + epsilon_32 * ln_ma^2 + epsilon_33 * ln_ma^3

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps30 : array
            epsilon30 array.
        eps31 : array
            epsilon31 array.
        eps32 : array
            epsilon32 array.
        eps33 : array
            epsilon33 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        _, eps_20, eps_21, eps_22 = self._eps_2(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_30 = (
            Phi_z * (eps_0 - ln_Mvir_z) ** 3 * zeta_z**3 / 6.0
            + Phi_z * (eps_0 - ln_Mvir_z) * eps_10 * (zeta_z**2)
            + Phi_z * eps_20 * zeta_z
        )
        integrand_31 = (
            Phi_z * (eps_0 - ln_Mvir_z) ** 2 * (zeta_z**3) / 2.0
            + Phi_z * eps_10 * (zeta_z**2)
            + Phi_z * eps_21 * zeta_z
            + Phi_z * (eps_0 - ln_Mvir_z) * eps_11 * (zeta_z**2)
        )
        integrand_32 = (
            Phi_z * (eps_0 - ln_Mvir_z) * (zeta_z**3) / 2.0
            + Phi_z * eps_11 * (zeta_z**2)
            + Phi_z * eps_22 * zeta_z
        )
        integrand_33 = Phi_z * (zeta_z**3) / 6.0
        integral_30 = cumulative_trapezoid(integrand_30, x=_z, initial=0)
        integral_31 = cumulative_trapezoid(integrand_31, x=_z, initial=0)
        integral_32 = cumulative_trapezoid(integrand_32, x=_z, initial=0)
        integral_33 = cumulative_trapezoid(integrand_33, x=_z, initial=0)
        return _z, integral_30, integral_31, integral_32, integral_33

    def subhalo_mass_stripped_pert0(self, ma, za, z):
        r"""Calculate subhalo mass stripping using zeroth-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-0 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        eps_0 = self.eps_0(za, z)
        return ma * np.exp(eps_0)

    def subhalo_mass_stripped_pert1(self, ma, za, z):
        r"""Calculate subhalo mass stripping using first-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-1 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        ln_ma = np.log(ma)
        eps = eps_0 + eps_10 + ln_ma * eps_11
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert2(self, ma, za, z):
        r"""Calculate subhalo mass stripping using second-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-2 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        eps = (
            eps_0
            + eps_10
            + ln_ma * eps_11
            + eps_20
            + ln_ma * eps_21
            + ln_ma**2 * eps_22
        )
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert3(self, ma, za, z):
        r"""Calculate subhalo mass stripping using third-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-3 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        eps_30 = self.eps_30(za, z)
        eps_31 = self.eps_31(za, z)
        eps_32 = self.eps_32(za, z)
        eps_33 = self.eps_33(za, z)
        ln_ma = np.log(ma)
        eps = (
            eps_0
            + eps_10
            + ln_ma * eps_11
            + eps_20
            + ln_ma * eps_21
            + ln_ma**2 * eps_22
            + eps_30
            + ln_ma * eps_31
            + ln_ma**2 * eps_32
            + ln_ma**3 * eps_33
        )
        return ma * np.exp(eps)

    def subhalo_mass_stripped(self, ma, za, z, method="picard_table", **kwargs):
        r"""A wrapper function to calculate subhalo mass stripping.

        Parameters
        ----------
        ma : float
            initial subhalo mass.
        za : float
            initial redshift.
        z : float
            final redshift.
        method : str
            Explicit tidal integration method: picard_table, dop853, odeint or a retained
            perturbative approximation.
        **kwargs : dict
            Controls for the selected Picard or ODE backend. Historical odeint Dfun callbacks
            use state-first signature; unsupported controller overrides are rejected.

        Returns
        -------
        float or numpy.ndarray
            Final bound masses in the ma unit; this endpoint interface does not return a
            redshift/mass tuple.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """
        match method:
            case "picard_table":
                return self.subhalo_mass_stripped_picard_table(ma, za, z, **kwargs)
            case "dop853":
                return get_picard_solver(self).direct(ma, za, z, **kwargs)
            case "odeint":
                return self.subhalo_mass_stripped_odeint(ma, za, z, **kwargs)
            case "pert0":
                return self.subhalo_mass_stripped_pert0(ma, za, z)
            case "pert1":
                return self.subhalo_mass_stripped_pert1(ma, za, z)
            case "pert2":
                return self.subhalo_mass_stripped_pert2(ma, za, z)
            case "pert2_shanks":
                return self.subhalo_mass_stripped_pert2_shanks(ma, za, z)
            case "pert3":
                return self.subhalo_mass_stripped_pert3(ma, za, z)
            case _:
                raise ValueError(f"Invalid method: {method}")


class CDMObservableKernels:
    r"""Observable reductions over a previously computed named population.

    Notes
    -----
    These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
    base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
    observable boundaries. The fitted relations and mass definitions remain model choices,
    not consequences of the shared numerical executor.
    """

    def mass_function(self, evolved=True):
        r"""Histogram the selected surviving population in present or accretion mass.

        Parameters
        ----------
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.

        Returns
        -------
        tuple of numpy.ndarray
            Mass-bin centers in Msun and dN/dln(m) from 100 logarithmic histogram bins.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        if evolved:
            N, lnm_edges = np.histogram(np.log(self.m0), weights=self.weight, bins=100)
        else:
            N, lnm_edges = np.histogram(
                np.log(self.ma200), weights=self.weight, bins=100
            )

        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        dlnm = lnm_edges[1:] - lnm_edges[:-1]

        m = np.exp(lnm)
        dNdlnm = N / dlnm

        return m / self.Msun, dNdlnm

    def Nsat_Mpeak(self, Mpeak_th):
        r"""Count surviving subhalos passing the accretion-mass satellite proxy threshold.

        Parameters
        ----------
        Mpeak_th : float
            Satellite proxy threshold on accretion M200, in the internal mass unit; use the
            documented strict comparison.

        Returns
        -------
        tuple of numpy.ndarray
            Present mass centers (Msun), complementary mass counts, Vmax centers (km/s), and
            complementary velocity counts.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        N, lnm_edges = np.histogram(
            np.log(self.m0[self.ma200 > Mpeak_th]),
            weights=self.weight[self.ma200 > Mpeak_th],
            bins=100,
        )
        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        m = np.exp(lnm)
        Ncum = np.cumsum(N)
        Nccum_m = Ncum[-1] - Ncum

        N, lnVmax_edges = np.histogram(
            np.log(self.Vmax[self.ma200 > Mpeak_th]),
            weights=self.weight[self.ma200 > Mpeak_th],
            bins=100,
        )
        lnVmax = (lnVmax_edges[1:] + lnVmax_edges[:-1]) / 2.0
        Vmax = np.exp(lnVmax)
        Ncum = np.cumsum(N)
        Nccum_Vmax = Ncum[-1] - Ncum

        return m / self.Msun, Nccum_m, Vmax / (self.km / self.s), Nccum_Vmax

    def Nsat_Vpeak(self, Vpeak_th):
        r"""Count surviving subhalos passing the peak-speed satellite proxy threshold.

        Parameters
        ----------
        Vpeak_th : float
            Satellite proxy threshold on peak circular speed, in internal velocity units; for
            km/s multiply by model.km/model.s.

        Returns
        -------
        tuple of numpy.ndarray
            Present mass centers (Msun), complementary mass counts, Vmax centers (km/s), and
            complementary velocity counts.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        N, lnm_edges = np.histogram(
            np.log(self.m0[self.Vpeak > Vpeak_th]),
            weights=self.weight[self.Vpeak > Vpeak_th],
            bins=100,
        )
        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        m = np.exp(lnm)
        Ncum = np.cumsum(N)
        Nccum_m = Ncum[-1] - Ncum

        N, lnVmax_edges = np.histogram(
            np.log(self.Vmax[self.Vpeak > Vpeak_th]),
            weights=self.weight[self.Vpeak > Vpeak_th],
            bins=100,
        )
        lnVmax = (lnVmax_edges[1:] + lnVmax_edges[:-1]) / 2.0
        Vmax = np.exp(lnVmax)
        Ncum = np.cumsum(N)
        Nccum_Vmax = Ncum[-1] - Ncum

        return m / self.Msun, Nccum_m, Vmax / (self.km / self.s), Nccum_Vmax

    def mass_fraction(self, evolved=True):
        r"""Sum the selected subhalo masses relative to the host M200 at the target epoch.

        Parameters
        ----------
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.

        Returns
        -------
        float
            Dimensionless weighted subhalo mass fraction; evolved chooses bound versus accretion
            mass.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        Mhost = self.Mzi(self.M0, self.redshift)
        if evolved:
            fsh = np.sum(self.m0 * self.weight) / Mhost
        else:
            fsh = np.sum(self.ma200 * self.weight) / Mhost

        return fsh

    def annihilation_boost_factor(
        self, n=0, evolved=True, *, allow_incomplete_tables=False
    ):
        r"""Integrate the NFW annihilation luminosity of subhalos and optional recursive
        substructure.

        Parameters
        ----------
        n : int
            Nonnegative substructure hierarchy order. Zero uses the direct subhalo contribution;
            higher orders require previous-order tables.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        allow_incomplete_tables : bool
            Explicit opt-in to the historical undefined-interpolation-to-zero approximation,
            with invalid-point diagnostics. It is not a calibrated low-mass boundary.

        Returns
        -------
        tuple of float
            B_sh and total-to-smooth-host luminosity ratio 1-f_sh**2+B_sh, both dimensionless.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        validate_order(n)
        fsh = self.mass_fraction(evolved)
        if n == 0:
            fssh = 0.0
            Bssh = 0.0
        else:
            tables = interpolate_boost_tables(
                self,
                getattr(self, "_boost_table_directory", self.data_dir / "boost"),
                {"Bssh": (f"Bsh_{n - 1}.txt", 1.0e-30), "fssh": ("fsh.txt", 0.0)},
                allow_incomplete_tables=allow_incomplete_tables,
            )
            Bssh, fssh = tables["Bssh"], tables["fssh"]

            mavir = self.Mvir_from_M200_fit(self.ma200, self.z_a)
            Oz = self.OmegaM * (1.0 + self.z_a) ** 3 / self.g(self.z_a)
            ravir = np.cbrt(
                3.0
                * mavir
                / (4.0 * np.pi * self.rhocrit(self.z_a) * self.Delc(Oz - 1.0))
            )
            cavir = ravir / self.rs_a

            Bssh = (
                Bssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            Bssh = Bssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            Bssh = Bssh / (
                self.rhos0**2 * self.rs0**3 * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
            )
            Bssh = Bssh * (
                self.rhos_a**2 * self.rs_a**3 * (1.0 - 1.0 / (1.0 + cavir) ** 3)
            )
            fssh = (
                fssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            fssh = fssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            fssh = fssh / (self.rhos0 * self.rs0**3 * self.fc(self.ct0))
            fssh = fssh * (self.rhos_a * self.rs_a**3 * self.fc(cavir))

        if evolved:
            Lsh = np.sum(
                (1.0 - fssh**2 + Bssh)
                * self.rhos0**2
                * self.rs0**3
                * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
                * self.weight
            )
        else:
            r200 = (
                3.0 * self.ma200 / (4.0 * np.pi * self.rhocrit(self.redshift) * 200.0)
            ) ** (1.0 / 3.0)
            c200 = r200 / self.rs_a
            Lsh = np.sum(
                self.rhos_a**2
                * self.rs_a**3
                * (1.0 - 1.0 / (1.0 + c200) ** 3)
                * self.weight
            )

        Mhost = self.Mzi(self.M0, self.redshift)
        r200_host = (
            3.0 * Mhost / (4.0 * np.pi * self.rhocrit(self.redshift) * 200.0)
        ) ** (1.0 / 3.0)
        c200_host = self.conc200(Mhost, self.redshift)
        rs_host = r200_host / c200_host
        rhos_host = Mhost / (4.0 * np.pi * rs_host**3 * self.fc(c200_host))
        Lhost0 = rhos_host**2 * rs_host**3 * (1.0 - 1.0 / (1.0 + c200_host) ** 3)

        Bsh = Lsh / Lhost0
        luminosity_ratio = 1.0 - fsh**2 + Bsh

        return Bsh, luminosity_ratio

    def annihilation_boost_factor_prompt_cusps(
        self, n=0, f_surv=1.0, f_surv_stripped=1.0, *, allow_incomplete_tables=False
    ):
        r"""Add bound and stripped prompt-cusp luminosities to the recursive subhalo boost.

        Parameters
        ----------
        n : int
            Nonnegative substructure hierarchy order. Zero uses the direct subhalo contribution;
            higher orders require previous-order tables.
        f_surv : float
            Dimensionless surviving fraction for prompt cusps in the bound or stripped
            component, as selected by the named argument.
        f_surv_stripped : float
            Dimensionless surviving fraction for prompt cusps in the bound or stripped
            component, as selected by the named argument.
        allow_incomplete_tables : bool
            Explicit opt-in to the historical undefined-interpolation-to-zero approximation,
            with invalid-point diagnostics. It is not a calibrated low-mass boundary.

        Returns
        -------
        tuple
            B_sh, dressed-cusp boost, naked-cusp boost, total luminosity ratio, dressed-cusp
            count and naked-cusp count.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        validate_order(n)
        from .prompt_cusps import prompt_cusps as _prompt_cusps

        prc = _prompt_cusps(k_fs=self.k_fs, data_dir=self.data_dir)
        _f_coll, J_cusps = prc.cusp_properties(f_surv=1.0, z=self.redshift)
        J_cusps_mean = np.mean(J_cusps)

        fsh = self.mass_fraction()
        if n == 0:
            fssh = 0.0
            Bssh = 0.0
            Ncusp_dressed = 0.0
            Ncusp_naked = 0.0
        else:
            suffix = f"{n - 1}_{f_surv:.1f}_{f_surv_stripped:.1f}.txt"
            tables = interpolate_boost_tables(
                self,
                getattr(
                    self,
                    "_boost_table_directory",
                    self.data_dir / "prompt_cusps" / "boost",
                ),
                {
                    "Bssh": (f"Bsh_{suffix}", 1.0e-30),
                    "fssh": ("fsh.txt", 0.0),
                    "Ncusp_dressed": (f"Ncusp_dressed_{suffix}", 0.0),
                    "Ncusp_naked": (f"Ncusp_naked_{suffix}", 0.0),
                },
                allow_incomplete_tables=allow_incomplete_tables,
                extra_contract={
                    "f_surv": float(f_surv),
                    "f_surv_stripped": float(f_surv_stripped),
                },
            )
            Bssh, fssh = tables["Bssh"], tables["fssh"]
            Ncusp_dressed0, Ncusp_naked0 = (
                tables["Ncusp_dressed"],
                tables["Ncusp_naked"],
            )

            mavir = self.Mvir_from_M200_fit(self.ma200, self.z_a)
            Oz = self.OmegaM * (1.0 + self.z_a) ** 3 / self.g(self.z_a)
            ravir = np.cbrt(
                3.0
                * mavir
                / (4.0 * np.pi * self.rhocrit(self.z_a) * self.Delc(Oz - 1.0))
            )
            cavir = ravir / self.rs_a

            fssh = (
                fssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            fssh = fssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            fssh = fssh / (self.rhos0 * self.rs0**3 * self.fc(self.ct0))
            fssh = fssh * (self.rhos_a * self.rs_a**3 * self.fc(cavir))

            Bssh = (
                Bssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            Bssh = Bssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            Bssh = Bssh / (
                self.rhos0**2 * self.rs0**3 * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
            )
            Bssh = Bssh * (
                self.rhos_a**2 * self.rs_a**3 * (1.0 - 1.0 / (1.0 + cavir) ** 3)
            )

            Ncusp_dressed = (
                Ncusp_dressed0
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
                / (self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2)))
            )
            Ncusp_naked = Ncusp_naked0 + f_surv_stripped * (
                Ncusp_dressed0 - Ncusp_dressed
            )

        Ncusp_dressed = f_surv * np.sum(self.weight) + np.sum(
            Ncusp_dressed * self.weight
        )
        Ncusp_naked = np.sum(Ncusp_naked * self.weight)

        Lsh = np.sum(
            (1.0 - fssh**2 + Bssh)
            * 4.0
            * np.pi
            / 3.0
            * self.rhos0**2
            * self.rs0**3
            * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
            * self.weight
        )
        Lcusp_dressed = J_cusps_mean * Ncusp_dressed
        Lcusp_naked = J_cusps_mean * Ncusp_naked

        Mhost = self.Mzi(self.M0, self.redshift)
        r200_host = (
            3.0 * Mhost / (4.0 * np.pi * self.rhocrit(self.redshift) * 200.0)
        ) ** (1.0 / 3.0)
        c200_host = self.conc200(Mhost, self.redshift)
        rs_host = r200_host / c200_host
        rhos_host = Mhost / (4.0 * np.pi * rs_host**3 * self.fc(c200_host))
        Lhost0 = (
            4.0
            * np.pi
            / 3.0
            * rhos_host**2
            * rs_host**3
            * (1.0 - 1.0 / (1.0 + c200_host) ** 3)
        )

        Bsh = Lsh / Lhost0
        Bcusp_dressed = Lcusp_dressed / Lhost0
        Bcusp_naked = Lcusp_naked / Lhost0
        luminosity_ratio = 1.0 - fsh**2 + Bsh + Bcusp_dressed + Bcusp_naked

        return (
            Bsh,
            Bcusp_dressed,
            Bcusp_naked,
            luminosity_ratio,
            Ncusp_dressed,
            Ncusp_naked,
        )

    def subhalo_catalog_MC(self, mth, *, rng=None, seed=None):
        r"""Draw independent Poisson multiplicities for surviving nodes above a bound-mass
        threshold.

        Parameters
        ----------
        mth : float
            Nonnegative bound-mass threshold for the sampled catalogue, in the internal mass
            unit.
        rng : numpy.random.Generator
            Explicit generator used to draw independent Poisson multiplicities from the
            deterministic expected counts.
        seed : int or None
            Seed for a local random generator; mutually exclusive with an explicitly supplied
            rng.

        Returns
        -------
        tuple of numpy.ndarray
            Eight realized columns: accretion mass (Msun), redshift, accretion r_s (kpc), rho_s
            (Msun/pc^3), bound mass (Msun), present r_s (kpc), rho_s (Msun/pc^3), and c_t.

        Notes
        -----
        These retained CDM prescriptions are evaluated with Omega_m0=0.315 and h=0.674; their original fit calibrations are not changed. Internal
        base units are Msun, Mpc and seconds; internal velocities are Mpc/s, converted at
        observable boundaries. The fitted relations and mass definitions remain model choices,
        not consequences of the shared numerical executor.
        """

        if not np.isfinite(mth) or mth < 0:
            raise ValueError("Mass threshold must be finite and nonnegative.")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)
        condition = self.m0 > mth
        columns = {
            "ma200": self.ma200[condition],
            "z_a": self.z_a[condition],
            "rs_a": self.rs_a[condition],
            "rhos_a": self.rhos_a[condition],
            "m0": self.m0[condition],
            "rs0": self.rs0[condition],
            "rhos0": self.rhos0[condition],
            "ct0": self.ct0[condition],
        }
        population = WeightedSubhaloCatalog(
            columns=columns,
            weights={"weight_base": self.weight[condition]},
            metadata={
                "schema_version": "1.0",
                "model_identifier": "sashimi-c:mc:v1",
                "backend_identifier": self.itamae_cosmology.identifier,
            },
        )
        realization = population.poisson_realization(rng)
        ma200_MC, z_a_MC = realization["ma200"], realization["z_a"]
        rs_a_MC, rhos_a_MC = realization["rs_a"], realization["rhos_a"]
        m0_MC, rs0_MC = realization["m0"], realization["rs0"]
        rhos0_MC, ct0_MC = realization["rhos0"], realization["ct0"]

        return (
            ma200_MC / self.Msun,
            z_a_MC,
            rs_a_MC / self.kpc,
            rhos_a_MC / (self.Msun / self.pc**3),
            m0_MC / self.Msun,
            rs0_MC / self.kpc,
            rhos0_MC / (self.Msun / self.pc**3),
            ct0_MC,
        )
