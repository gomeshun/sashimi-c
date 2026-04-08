import unittest
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, vmap

import sashimi_c as v1
import sashimi_c_v2 as v2
from numpy.polynomial.hermite import hermgauss



class TestUnitAndConstants(unittest.TestCase):

    def test_units_and_constants(self):
        """ Check units_and_constants.py matches v1 """
        for attr in dir(v1.units_and_constants()):
            if not attr.startswith('_'):
                val1 = getattr(v1.units_and_constants(), attr)
                val2 = getattr(v2.units_and_constants(), attr)
                self.assertEqual(val1, val2, f"Mismatch in units_and_constants for {attr}")
                val3 = getattr(v2.units_and_constants, attr)
                self.assertEqual(val1, val3, f"Mismatch in units_and_constants_v2 for {attr}")

class TestHaloModel(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.v1 = v1.halo_model()
        self.v2 = v2.halo_model()

    def test_dsdm(self):
        """ Check dsdm matches between v1 and v2 """
        M = np.logspace(6, 15)
        z = 0.0
        M,z = jnp.broadcast_arrays(M,z)
        dsdm_v1 = self.v1.dsdm(M,z)
        dsdm_v2 = vmap(self.v2.dsdm_jax)(M, z)
        unittest.TestCase.assertTrue(
            self,
            np.allclose(dsdm_v1, dsdm_v2, rtol=1e-5),
            "Mismatch in dsdm between v1 and v2"
        )

class TestSubhaloProperties(unittest.TestCase):

    def setUp(self):
        self.v1 = v1.subhalo_properties()
        self.v2 = v2.subhalo_properties()

    def test_subhalo_observables(self):
        M0 = 1e12
        ret_v1 = self.v1.subhalo_properties_calc(M0)
        ret_v2 = self.v2.subhalo_properties_calc(M0)

        # JAX defaults (esp. float32 on GPU) can introduce small-but-noticeable
        # numerical differences vs the NumPy (v1) implementation.
        names = [
            'ma200','z_acc','rs_acc','rhos_acc','m_z0','rs_z0','rhos_z0','ct_z0','weight','survive'
        ]
        for name, item_v1, item_v2 in zip(names, ret_v1, ret_v2):
            a = np.asarray(item_v1)
            b = np.asarray(item_v2)
            if name == 'survive':
                self.assertTrue(np.array_equal(a.astype(bool), b.astype(bool)), f"Mismatch in {name}")
                continue

            rtol = 1e-3
            if name == 'weight':
                rtol = 1.5e-1
            np.testing.assert_allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True, err_msg=f"Mismatch in {name}")

if __name__ == '__main__':
    unittest.main()