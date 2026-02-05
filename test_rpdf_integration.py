import numpy as np
from sashimi_c import subhalo_properties

def test_subhalo_rpdf_basic():
    s = subhalo_properties()
    out = s.subhalo_properties_r_dependence_calc(1e12, redshift=0.0, dz=1.0, zmax=2.0, N_ma=4, N_herm=2,
                                                  q_bin=8, orbit_samples=64, orbit_n_theta=32)
    # expected number of returned entries (same as before)
    assert len(out) >= 12
    ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0, ct_z0, weight, density, survive, q = out[:12]
    # lengths must match
    n = len(ma200)
    assert all(len(x) == n for x in (z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0, ct_z0, weight, density, survive, q))
    # q values must be finite and within [0,1]
    q = np.asarray(q)
    assert np.isfinite(q).all()
    assert (q >= 0).all() and (q <= 1.0 + 1e-12).all()
