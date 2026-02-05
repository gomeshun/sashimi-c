import unittest
import numpy as np
import matplotlib.pyplot as plt

import sashimi_c

from logging import getLogger, StreamHandler, DEBUG, FileHandler, Formatter
import datetime
from pathlib import Path

# Utility functions
def einasto(r,alpha = 0.678 , r_2 = 0.81,N=1):
    """ Einasto profile 
    
    Parameters
    ----------
    r : float or array-like
        Radius (kpc)
    alpha : float, optional
        Shape parameter, by default 0.678
    r_2 : float, optional
        Scale radius (kpc), by default 0.81
    N : float, optional
        Normalization, by default 1"""
    power = (-2/alpha)*((r/r_2)**alpha-1)
    return N*np.exp(power)  

class TestRDependent(unittest.TestCase):
    # Ensure linters know this attribute exists
    output_dir: Path

    def setUp(self):
        self.sh = sashimi_c.subhalo_properties()
        self.logger = getLogger(__name__)
        self.logger.setLevel(DEBUG)
        # Ensure output directory exists (project_root/log)
        self.output_dir = Path(__file__).resolve().parent / "log"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set up console logging
        stream_handler = StreamHandler()
        stream_handler.setLevel(DEBUG)
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        # Initialize/rotate file handler with timestamped filename
        self.update_file_handler_with_timestamp(base_filename='test_r_dep', ext='log')
    
    def update_file_handler_with_timestamp(self, base_filename: str = 'log', ext: str = 'log') -> str:
        """Replace existing file handler(s) with a new one whose filename includes current timestamp.

        Parameters
        ----------
        base_filename : str
            Prefix for the log file name (without extension).
        ext : str
            File extension. Defaults to 'log'.

        Returns
        -------
        str
            The new log file name.
        """
        # Compute timestamp and new filename
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_log = f"{base_filename}_{ts}.{ext}" if base_filename else f"{ts}.{ext}"

        # Remove and close any existing FileHandler(s) to avoid duplicates and file locks
        for h in list(self.logger.handlers):
            if isinstance(h, FileHandler):
                self.logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        # Reuse existing formatter if available from any remaining handler
        formatter = None
        for h in self.logger.handlers:
            if getattr(h, 'formatter', None) is not None:
                formatter = h.formatter
                break
        if formatter is None:
            formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create and attach new FileHandler (inside ./log directory)
        log_path = self.output_dir / new_log
        fh = FileHandler(log_path)
        fh.setLevel(DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Keep now in sync for any file outputs that also use the timestamp
        self.now = ts

        self.logger.info(f"Logging to file: {log_path}")
        return new_log
    

    # plot density profile
    def _test_plot_hist(self, q_bin,bins,m0_min, m0_max,
                mdot_fitting_type=0,
                A=0.85,alpha=1.8):
        results = self.sh.subhalo_properties_r_dependence_calc(
            1e12,q_bin=q_bin,
            dz=0.01,
            logmamin=-2,
            N_herm=5,
            mdot_fitting_type=mdot_fitting_type,
            A=A,alpha=alpha)
        # unpack results
        (
            ma200, 
            z_acc, 
            rs_acc, 
            rhos_acc, 
            m_z0, 
            rs_z0, 
            rhos_z0, 
            ct_z0, 
            weight_combined, 
            density, 
            survive, 
            q
        ) = results
        q = q.reshape(-1)
        w = weight_combined.reshape(-1)
        m0 = np.array(m_z0).reshape(-1)
        z_acc = np.array(z_acc).reshape(-1)
        mask = np.array(survive).reshape(-1)  # survive
        mask = mask & np.array(m0 > m0_min).reshape(-1)   # only plot for m0 > m0_min
        mask = mask & np.array(m0 < m0_max).reshape(-1)
        # print('mask', mask.sum(), 'out of', mask.size)
        self.logger.debug(f'mask {mask.sum()} out of {mask.size}')
        self.logger.debug(f'q: {q[mask]}')
        self.logger.debug(f'weight: {w[mask]}')

        # Create a single figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

        # (1) Weighted histogram of q
        hist_ret = axes[0].hist(
            q[mask],
            weights=w[mask],
            bins=bins,
            histtype='step',
        )
        axes[0].set_xlabel('q')
        axes[0].set_ylabel('weighted count')
        axes[0].set_title('Weighted q histogram')

        # (2) 2D histogram: z_acc vs q
        im = axes[1].hist2d(
            q[mask],
            z_acc[mask],
            bins=[q_bin, 32],
            weights=w[mask],
            cmap='viridis'
        )
        fig.colorbar(im[3], ax=axes[1])
        axes[1].set_xlabel('q')
        axes[1].set_ylabel('z_acc')
        axes[1].set_title('z_acc vs q (weighted)')

        # (3) Normalized density profile vs q
        val = hist_ret[0]
        edges = hist_ret[1]
        centers = (edges[1:] + edges[:-1]) / 2
        dq = edges[1:] - edges[:-1]
        val = val/(4*np.pi*centers**2*dq)
        val = val / np.exp(np.log(val.mean()))
        axes[2].loglog(centers, val, 'o-', label='simulation')
        NFW = lambda x: x**-1 * (1/7+x)**-2
        val_nfw = NFW(centers)
        val_nfw = val_nfw / np.exp(np.log(val_nfw.mean()))
        val_einasto = einasto(centers)
        val_einasto = val_einasto / np.exp(np.log(val_einasto.mean()))
        axes[2].loglog(centers, val_nfw, 'o-', label='NFW')
        axes[2].loglog(centers, val_einasto, 'o-', label='Einasto')
        axes[2].set_xlabel('q')
        axes[2].set_ylabel('normalized density')
        axes[2].set_title('Density profile')
        axes[2].legend()

        # Overall title and single save
        fig.suptitle(
            f'Density profile m0>{m0_min}, m0<{m0_max}, mdot_type={mdot_fitting_type}, A={A}, alpha={alpha}',
            fontsize=10
        )
        fname = self.output_dir / f'combined_plots_{self.now}.png'
        fig.savefig(fname, dpi=300)
        self.logger.info(f'Figure saved to {fname}')
        plt.close(fig)


    @property
    def default_params(self):
        return {
            "q_bin": 32,
            "bins": np.linspace(0, 1, 32 + 1),
            "m0_min": 1e-1,
            "m0_max": 1e0,
            "mdot_fitting_type": 2,
        }

    
    def test_r_dependent_default_params(self):
        args = self.default_params
        # Rotate log and timestamp to generate distinct outputs
        self.update_file_handler_with_timestamp()
        self._test_plot_hist(**args)

    def test_r_dependent_A06172_alpha180172(self):
        args = self.default_params
        self.update_file_handler_with_timestamp()
        self._test_plot_hist(A=0.6172, alpha=18.0172, **args)
        
    def test_r_dependent_orbit_evolved_distribution(self):
        """Run a small orbit-evolved calculation and save a q-distribution plot.
        This exercises the orbit-P(q|z_acc->z0) machinery added to
        `subhalo_properties_r_dependence_calc` and saves a plot to the test log dir.
        """
        self.update_file_handler_with_timestamp()
        # keep the run small so CI stays fast
        results = self.sh.subhalo_properties_r_dependence_calc(
            1e12,
            q_bin=32,
            redshift=0.0,
            dz=0.1,
            zmax=1.0,
            N_ma=200,
            N_herm=3,
            logmamin=-3,
            N_hermNa=50,
            mdot_fitting_type=2,
            q_max=1.0,
            orbit_samples=256,
            orbit_n_theta=64,
            orbit_seed=1,
            orbit_sign_mode='in'
        )

        (
            ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0,
            ct_z0, weight_combined, density, survive, q
        ) = results[:12]

        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        q = np.asarray(q).reshape(-1)
        w = np.asarray(weight_combined).reshape(-1)
        mask = np.asarray(survive).reshape(-1)

        # Basic sanity checks
        assert q.size == w.size
        assert np.isfinite(q).all()
        assert (q >= 0).all() and (q <= 1.0 + 1e-12).all()

        # Make and save a diagnostic plot
        bins = np.linspace(0, 1, 32 + 1).tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(q[mask], bins=bins, weights=w[mask], histtype='stepfilled', alpha=0.8)
        ax.set_xlabel('q')
        ax.set_ylabel('weighted count')
        ax.set_title('Orbit-evolved q distribution (test)')
        fname = Path(self.output_dir) / f'orbitevolved_{self.now}.png'
        fig.savefig(fname, dpi=200)
        plt.close(fig)

        # Ensure the file was created
        assert fname.exists()

    def test_r_dependent_production_parameters(self):
        """Run a production-scale orbit-evolved calculation and write QC plots.

        This uses the notebook's typical parameters: fine dz, full zmax, large N_ma
        and a reasonably large orbit_samples.  The test is intentionally long and
        writes plots to the test log directory; run it manually when you want full
        validation (or set RUN_LONG_TESTS=1 in the environment to enable in CI).
        """
        import os
        # Allow opt-in via environment variable to avoid accidental CI runs
        # if os.environ.get('RUN_LONG_TESTS','0') != '1':
        #     import pytest
        #     pytest.skip('Long production test skipped (set RUN_LONG_TESTS=1 to enable)')

        self.update_file_handler_with_timestamp()

        # Production-like parameters (be prepared: this is expensive!)
        results = self.sh.subhalo_properties_r_dependence_calc(
            1e12,
            q_bin=64,
            redshift=0.0,
            dz=0.01,
            zmax=7.0,
            N_ma=500,
            N_herm=5,
            logmamin=-6,
            N_hermNa=200,
            mdot_fitting_type=2,
            q_max=1.0,
            orbit_samples=2048,
            orbit_n_theta=256,
            orbit_seed=123,
            orbit_sign_mode='in'
        )

        (
            ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0,
            ct_z0, weight_combined, density, survive, q
        ) = results[:12]

        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        q = np.asarray(q).reshape(-1)
        w = np.asarray(weight_combined).reshape(-1)
        m0 = np.asarray(m_z0).reshape(-1)

        # Select only surviving subhalos within a fiducial mass range (same as notebook)
        m0_min = 1e-1
        m0_max = 1e0
        mask = (np.asarray(survive).reshape(-1)) & (m0 > m0_min) & (m0 < m0_max)

        # Basic checks
        assert q.size == w.size
        assert np.isfinite(q).all()
        assert (q >= 0).all() and (q <= 1.0 + 1e-12).all()
        assert w[mask].sum() > 0, 'No surviving subhalos in the selected mass range; consider adjusting m0_min/m0_max'
        # Save two diagnostic plots: total q-dist and z_acc vs q 2D map
        outdir = Path(self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 1, 64 + 1).tolist()
        ax.hist(q[mask], bins=bins, weights=w[mask], histtype='stepfilled', alpha=0.8)
        ax.set_xlabel('q')
        ax.set_ylabel('weighted count')
        ax.set_title('Orbit-evolved q distribution (production)')
        fname1 = outdir / f'orbitevolved_production_{self.now}.png'
        fig.savefig(fname1, dpi=200)
        self.logger.info(f"Figure saved to {fname1}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hist2d(q[mask], np.array(z_acc).reshape(-1)[mask], bins=[64, 128], weights=w[mask], cmap='viridis')
        fig.colorbar(hb[3], ax=ax)
        ax.set_xlabel('q')
        ax.set_ylabel('z_acc')
        ax.set_title('z_acc vs q (production)')
        fname2 = outdir / f'zacc_q_production_{self.now}.png'
        fig.savefig(fname2, dpi=200)
        self.logger.info(f"Figure saved to {fname2}")
        plt.close(fig)

        # -------------------------
        # Additional: 3D number density n(r) vs r, and overlay NFW/Einasto (same normalization)
        # -------------------------
        # Use q bins consistent with the q grid used for the run (64 bins)
        q_edges_nd = np.linspace(0, 1, 64 + 1)
        rvir_z0 = self.sh.host_virial_radius(self.sh._normalize_M0_to_z0(1e12, 0.0, False), 0.0)
        r_edges = q_edges_nd * rvir_z0
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_vol = (4.0 * np.pi / 3.0) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

        w_mask = w[mask]
        q_mask = q[mask]
        w_bin = np.histogram(q_mask, bins=q_edges_nd, weights=w_mask)[0]
        density_shell = np.where(shell_vol > 0, w_bin / shell_vol, 0.0)

        # Normalize the measured density for shape comparison (same as notebook)
        density_shell_norm = density_shell / np.exp(np.log(density_shell[density_shell>0].mean()))

        # Theoretical curves (evaluate on dimensionless q = r/Rvir)
        q_centers = 0.5 * (q_edges_nd[:-1] + q_edges_nd[1:])
        # NFW shape (not a physically normalized number density, only for shape comparison)
        NFW = lambda x: x**-1 * (1.0 / 7.0 + x)**-2
        val_nfw = NFW(q_centers)
        val_nfw = val_nfw / np.exp(np.log(val_nfw[val_nfw>0].mean()))

        # Einasto function (same as in notebook)
        def einasto_func(r, alpha=0.678, r_2=0.81, N=1.0):
            power = (-2.0/alpha)*((r/r_2)**alpha-1.0)
            return N*np.exp(power)

        val_einasto = einasto_func(q_centers)
        val_einasto = val_einasto / np.exp(np.log(val_einasto[val_einasto>0].mean()))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(r_centers, density_shell_norm, 'o-', label='simulation')
        ax.loglog(r_centers, val_nfw, '--', label='NFW (shape)')
        ax.loglog(r_centers, val_einasto, '-.', label='Einasto (shape)')
        ax.set_xlabel('r [Mpc]')
        ax.set_ylabel('normalized n(r)')
        ax.set_title('Orbit-evolved 3D number density n(r) (shape comparison)')
        ax.legend()
        fname3 = outdir / f'n_of_r_production_{self.now}.png'
        fig.savefig(fname3, dpi=200)
        self.logger.info(f"Figure saved to {fname3}")
        plt.close(fig)

        assert fname1.exists() and fname2.exists() and fname3.exists()

    def test_r_dependent_production_mass_bins_overlay(self):
        """Overlay n(r) for decade subhalo-mass bins on one plot.

        This answers the question "how does the spatial distribution depend on subhalo mass?" by
        selecting surviving subhalos in physical mass ranges (Msun) and plotting their 3D number
        density profiles n(r) on the same axes.

        Notes
        -----
        - This is intentionally expensive; enable it by setting RUN_LONG_TESTS=1.
        - Mass bins: 1e5–1e6, 1e6–1e7, ..., 1e9–1e10 Msun.
        """
        import os
        if os.environ.get('RUN_LONG_TESTS', '0') != '1':
            self.skipTest('Long production test skipped (set RUN_LONG_TESTS=1 to enable)')

        self.update_file_handler_with_timestamp()

        results = self.sh.subhalo_properties_r_dependence_calc(
            1e12,
            q_bin=64,
            redshift=0.0,
            dz=0.01,
            zmax=7.0,
            N_ma=500,
            N_herm=5,
            logmamin=-6,
            N_hermNa=200,
            mdot_fitting_type=2,
            q_max=1.0,
            orbit_samples=2048,
            orbit_n_theta=256,
            orbit_seed=123,
            orbit_sign_mode='in'
        )

        (
            ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0,
            ct_z0, weight_combined, density, survive, q
        ) = results[:12]

        q = np.asarray(q).reshape(-1)
        w = np.asarray(weight_combined).reshape(-1)
        m0 = np.asarray(m_z0).reshape(-1)
        survive_mask = np.asarray(survive).reshape(-1)

        assert q.size == w.size == m0.size == survive_mask.size
        assert np.isfinite(q).all()
        assert (q >= 0).all() and (q <= 1.0 + 1e-12).all()
        assert np.isfinite(m0).all() and (m0 >= 0).all()

        # radial bins in physical units
        q_edges_nd = np.linspace(0, 1, 64 + 1)
        rvir_z0 = self.sh.host_virial_radius(self.sh._normalize_M0_to_z0(1e12, 0.0, False), 0.0)
        r_edges = q_edges_nd * rvir_z0
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_vol = (4.0 * np.pi / 3.0) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

        # decade mass bins (Msun)
        mass_edges = 10.0 ** np.arange(5, 11)  # 1e5 ... 1e10
        mass_bins = list(zip(mass_edges[:-1], mass_edges[1:]))

        fig, ax = plt.subplots(figsize=(7, 5))
        plotted_any = False
        for mmin, mmax in mass_bins:
            mask = survive_mask & (m0 >= mmin) & (m0 < mmax)
            w_sum = float(w[mask].sum())
            self.logger.info(f"mass bin [{mmin:.1e}, {mmax:.1e}) Msun: N_eff={w_sum:.3e}")
            if w_sum <= 0:
                continue

            w_bin = np.histogram(q[mask], bins=q_edges_nd, weights=w[mask])[0]
            n_of_r = np.where(shell_vol > 0, w_bin / shell_vol, 0.0)
            pos = n_of_r > 0
            if not np.any(pos):
                continue

            # normalize for shape comparison (geometric mean)
            n_norm = n_of_r / np.exp(np.log(n_of_r[pos]).mean())
            ax.loglog(r_centers[pos], n_norm[pos], 'o-', ms=3, lw=1, label=f"{mmin:.0e}–{mmax:.0e} Msun")
            plotted_any = True

        assert plotted_any, 'No mass bins had non-zero surviving weight; consider increasing resolution or adjusting bins.'

        # Overlay reference shapes on the same (dimensionless) x=q grid, but plot against r for convenience.
        q_centers = 0.5 * (q_edges_nd[:-1] + q_edges_nd[1:])
        NFW = lambda x: x**-1 * (1.0 / 7.0 + x)**-2
        val_nfw = NFW(q_centers)
        val_nfw = val_nfw / np.exp(np.log(val_nfw[val_nfw > 0]).mean())
        ax.loglog(r_centers, val_nfw, '--', lw=1, color='k', alpha=0.6, label='NFW (shape)')

        val_einasto = einasto(q_centers)
        val_einasto = val_einasto / np.exp(np.log(val_einasto[val_einasto > 0]).mean())
        ax.loglog(r_centers, val_einasto, '-.', lw=1, color='0.3', alpha=0.6, label='Einasto (shape)')

        ax.set_xlabel('r [Mpc]')
        ax.set_ylabel('normalized n(r)')
        ax.set_title('Orbit-evolved 3D number density n(r) by subhalo mass bin')
        ax.legend(fontsize=8)

        outdir = Path(self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = outdir / f'n_of_r_massbins_production_{self.now}.png'
        fig.savefig(fname, dpi=200)
        self.logger.info(f"Figure saved to {fname}")
        plt.close(fig)

        assert fname.exists()

    def test_r_dependent_production_mass_vs_r_heatmap(self):
        """Plot a 2D histogram (heatmap) of position r vs mass m for surviving subhalos.

        x-axis: r [Mpc] (log)
        y-axis: m0 [Msun] (log)

        This is a complementary diagnostic to the 1D n(r) overlays: even if the
        *normalized* profiles look similar by eye, this shows the joint distribution.

        Notes
        -----
        - Intentionally expensive; enable with RUN_LONG_TESTS=1.
        """
        import os
        if os.environ.get('RUN_LONG_TESTS', '0') != '1':
            self.skipTest('Long production test skipped (set RUN_LONG_TESTS=1 to enable)')

        self.update_file_handler_with_timestamp()

        results = self.sh.subhalo_properties_r_dependence_calc(
            1e12,
            q_bin=64,
            redshift=0.0,
            dz=0.01,
            zmax=7.0,
            N_ma=500,
            N_herm=5,
            logmamin=-6,
            N_hermNa=200,
            mdot_fitting_type=2,
            q_max=1.0,
            orbit_samples=2048,
            orbit_n_theta=256,
            orbit_seed=123,
            orbit_sign_mode='in'
        )

        (
            ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0,
            ct_z0, weight_combined, density, survive, q
        ) = results[:12]

        q = np.asarray(q).reshape(-1)
        w = np.asarray(weight_combined).reshape(-1)
        m0 = np.asarray(m_z0).reshape(-1)
        survive_mask = np.asarray(survive).reshape(-1)

        assert q.size == w.size == m0.size == survive_mask.size
        assert np.isfinite(q).all()
        assert (q >= 0).all() and (q <= 1.0 + 1e-12).all()

        # Convert q -> r [Mpc] at z0
        rvir_z0 = self.sh.host_virial_radius(self.sh._normalize_M0_to_z0(1e12, 0.0, False), 0.0)
        r = q * float(rvir_z0)

        # Keep only valid / physical points
        mask = (
            survive_mask
            & np.isfinite(r) & (r > 0)
            & np.isfinite(m0) & (m0 > 0)
            & np.isfinite(w) & (w >= 0)
        )
        assert float(w[mask].sum()) > 0, 'No surviving weight after masking; cannot build heatmap.'

        # Log-spaced bins
        # r-range: use the resolved q-grid range (avoid zero)
        r_min = float(np.min(r[mask]))
        r_max = float(np.max(r[mask]))
        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), 64)

        # mass-range: focus on the decade range used in the overlay test
        m_bins = np.logspace(5, 10, 64)  # 1e5 ... 1e10 Msun

        H, xedges, yedges = np.histogram2d(
            r[mask],
            m0[mask],
            bins=(r_bins.tolist(), m_bins.tolist()),
            weights=w[mask]
        )

        assert np.isfinite(H).all()
        assert H.sum() > 0, (
            '2D histogram is empty in the requested mass range 1e5–1e10 Msun; '
            'check m0 units or adjust m_bins.'
        )

        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(7, 5))
        # pcolormesh expects edges; H is (Nx-1, Ny-1)
        X, Y = np.meshgrid(xedges, yedges, indexing='ij')
        pcm = ax.pcolormesh(
            X,
            Y,
            H,
            shading='auto',
            norm=mcolors.LogNorm(vmin=max(1e-12, float(H[H > 0].min())), vmax=float(H.max())),
            cmap='viridis'
        )
        fig.colorbar(pcm, ax=ax, label='weighted count')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('r [Mpc]')
        ax.set_ylabel('m0 [Msun]')
        ax.set_title('Surviving subhalos: mass vs radius (orbit-evolved, weighted)')

        outdir = Path(self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = outdir / f'mass_vs_r_heatmap_production_{self.now}.png'
        fig.savefig(fname, dpi=200)
        self.logger.info(f"Figure saved to {fname}")
        plt.close(fig)

        assert fname.exists()
    def test_r_dependent_A085_alpha180172(self):
        args = self.default_params
        self.update_file_handler_with_timestamp()
        self._test_plot_hist(A=0.85, alpha=18.0172, **args)