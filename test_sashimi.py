import unittest
import numpy as np
import sashimi_c
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
from logging import getLogger, DEBUG, StreamHandler, Formatter


class TestSashimiC(unittest.TestCase):
    """Tests for `sashimi_c.py`."""

    ORIGINAL_MASS_FUNCTION_FNAME = "test_mass_function_mass_function_20251002_135326.gz"

    def setUp(self) -> None:
        self.obs = sashimi_c.subhalo_observables(
            M0_per_Msun=1.e12,
            N_herm=20,)
        self.logger = getLogger(__name__)
        self.logger.setLevel(DEBUG)
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.debug("Setup complete.")


    def fname(self, body, ext) -> str:
        """Generate a filename based on the test method name and current timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._testMethodName}_{body}_{timestamp}.{ext}"


    def test_mass_function_original(self):
        arr = np.loadtxt(self.ORIGINAL_MASS_FUNCTION_FNAME)
        m, dNdlnm = arr[:, 0], arr[:, 1]
        plt.figure(figsize=(6,5))
        plt.loglog(m, m*dNdlnm)
        plt.xlim(1.e-7,3.e11)
        plt.ylim(1.e7,1.e11)
        plt.xlabel(r'$m$ [$M_{\odot}$]')
        plt.ylabel(r'$m^{2}dN_{\rm sh}/dm$ [$M_{\odot}$]')
        plt.title(r'$M_{200}=10^{12}M_{\odot}$ (Original Data)')
        plot_fname = self.fname('mass_function_original', 'png')
        plt.savefig(plot_fname)
        plt.close()
        self.logger.info(f"Saved original mass function plot to {plot_fname}")
        

    def test_mass_function(self):
        """Test mass function."""
        m,dNdlnm = self.obs.mass_function()
        # save data
        arr = np.vstack((m, dNdlnm)).T
        arr_fname = self.fname('mass_function', 'gz')
        np.savetxt(arr_fname, arr)
        self.logger.info(f"Saved mass function data to {arr_fname}")
        # plot
        plt.figure(figsize=(6,5))
        plt.loglog(m,m*dNdlnm)
        plt.xlim(1.e-7,3.e11)
        plt.ylim(1.e7,1.e11)
        plt.xlabel(r'$m$ [$M_{\odot}$]')
        plt.ylabel(r'$m^{2}dN_{\rm sh}/dm$ [$M_{\odot}$]')
        plt.title(r'$M_{200}=10^{12}M_{\odot}$')
        plot_fname = self.fname('mass_function', 'png')
        plt.savefig(plot_fname)
        plt.close()
        self.logger.info(f"Saved mass function plot to {plot_fname}")

    
    def test_mass_function_compare(self):
        """Test mass function comparison to original."""
        m,dNdlnm = self.obs.mass_function()
        arr_new = np.vstack((m, dNdlnm)).T
        arr_orig = np.loadtxt(self.ORIGINAL_MASS_FUNCTION_FNAME)
        np.testing.assert_allclose(arr_new, arr_orig, rtol=1.e-5)
        self.logger.info("Mass function matches original data within tolerance.")


    def test_vmax_rmax(self):
        sh = self.obs
        mth = 1.e8*sh.Msun/sh.h
        condition = (sh.m0>mth)*(sh.weight>0.)
        q1 = sh.Vmax[condition]/(sh.km/sh.s)
        q2 = sh.rmax[condition]/sh.kpc
        plt.figure(figsize=(6,5))
        d2Ndxdy,x_edges,y_edges = np.histogram2d(
            np.log(q1),
            np.log(q2),
            weights=sh.weight[condition],bins=[30,30],
            density=True)
        x = (x_edges[:-1]+x_edges[1:])/2.
        y = (y_edges[:-1]+y_edges[1:])/2.
        d2Ndxdy *= np.sum(sh.weight[condition])  # Convert to counts
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(5e0,1e2)
        plt.ylim(1e-1,5e1)
        plt.xlabel(r'$V_{\rm max} \ [\mathrm{km/s}]$')
        plt.ylabel(r'$r_{\rm max} \ [\mathrm{kpc}]$')
        plt.title(r'CDM ($m_{\rm sh}>10^{8}M_{\odot}/h$)')
        levels = np.linspace(-3,2,21)
        plt.contourf(np.exp(x),np.exp(y),np.log10(d2Ndxdy.T),levels=levels,cmap='RdBu_r')
        cbar = plt.colorbar(ticks=np.arange(2,-3.1,-1))
        cbar.set_label(r'$\log(d^{2}N_{\rm sh}/d\ln\,V_{\rm max}/d\ln\,r_{\rm max})$',fontsize=15)
        plot_fname = self.fname('vmax_rmax', 'png')
        plt.savefig(plot_fname)
        plt.close()
        self.logger.info(f"Saved Vmax-Rmax plot to {plot_fname}")


    def test_cumulative_number(self):
        m,Nccum_m,Vmax,Nccum_Vmax = self.obs.Nsat_Mpeak(1.e8*self.obs.Msun)
        plt.figure(figsize=(6,5))
        plt.loglog(m,Nccum_m)
        plt.xlabel(r'$m$ [$M_{\odot}$]')
        plt.ylabel(r'$N_{\rm sat}(>m)$')
        plt.title(r'$m_{\rm peak}>10^{8}M_{\odot}$')
        plt.savefig(self.fname('cumulative_number_mass', 'png'))
        plt.close()
        self.logger.info(f"Saved cumulative number (mass) plot to {self.fname('cumulative_number_mass', 'png')}")

        plt.figure(figsize=(6,5))
        plt.loglog(Vmax,Nccum_Vmax)
        plt.xlabel(r'$V_{\rm max}$ [km s$^{-1}$]')
        plt.ylabel(r'$N_{\rm sat}(>V_{\rm max})$')
        plt.title(r'$m_{\rm peak}>10^{8}M_{\odot}$')
        plt.savefig(self.fname('cumulative_number_vmax', 'png'))
        plt.close()
        self.logger.info(f"Saved cumulative number (vmax) plot to {self.fname('cumulative_number_vmax', 'png')}")

    def test_tidal_stripping_consistency(self):
        """Test consistency of TidalStripping equation solution methods."""
        from sashimi_c import TidalStrippingSolver
        # Follow main code usage: masses are in grams (Msun units), method returns final mass only
        # Choose a typical host and subhalo setup used elsewhere in the code
        M0_Msun = 1e12
        ma = 1e6  # Test multiple subhalo masses
        za = 7.0   # initial redshift
        z0 = 0.0   # final redshift
        solver = TidalStrippingSolver(M0_Msun * TidalStrippingSolver(1.0).Msun)

        # Build inputs (scalar and array) in grams
        Msun = solver.Msun
        ma_scalar = ma * Msun
        ma_array = np.logspace(0,9,16) * Msun

        methods = ["pert2_shanks", "odeint", "pert3", "pert2", "pert1", "pert0"]
        tolerances = {
            "odeint": 0.0,  # reference
            "pert2_shanks": 5e-3,
            "pert3": 2e-2,
            "pert2": 6e-2,
            "pert1": 5e-1,
            "pert0": 1e-0,
        }

        # Compute results for scalar input
        results_scalar = {}
        for method in methods:
            kwargs = dict(rtol=1e-12,atol=1e-12) if method == "odeint" else {}
            m_final = solver.subhalo_mass_stripped(ma_scalar, za, z0, method=method, **kwargs)
            results_scalar[method] = np.asarray(m_final)

        # Compute results for array input and assert shape preservation
        results_array = {}
        for method in methods:
            m_final = solver.subhalo_mass_stripped(ma_array, za, z0, method=method)
            m_final = np.asarray(m_final)
            self.assertEqual(m_final.shape, ma_array.shape, f"Output shape mismatch for {method}")
            results_array[method] = m_final

        # Use odeint as the reference
        ref_scalar = results_scalar["odeint"]
        ref_array = results_array["odeint"]

        # Compare each method to the reference with per-method tolerances
        for method in methods:
            rtol = tolerances[method]
            np.testing.assert_allclose(results_scalar[method], ref_scalar, rtol=rtol,
                                       err_msg=f"Scalar mismatch vs odeint for {method}")
            np.testing.assert_allclose(results_array[method], ref_array, rtol=rtol,
                                       err_msg=f"Array mismatch vs odeint for {method}")
            self.logger.info(f"max relative difference for {method}: {np.max(np.abs(results_array[method] - ref_array)/ref_array)}")
        self.logger.info("TidalStripping methods are consistent (scalar and array inputs) within method-specific tolerances.")