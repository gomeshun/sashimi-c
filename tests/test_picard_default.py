import inspect

import numpy as np
from scipy.integrate import solve_ivp

from sashimi_c import TidalStrippingSolver, subhalo_observables, subhalo_properties


def reference_log_ode_mass(solver, ma, z_acc, z_final=0.0):
    def rhs(z, y):
        return [
            solver.Phi(z)
            * np.exp(solver.zetaMz(z) * (y[0] - np.log(solver.Mzvir(z))))
        ]

    result = solve_ivp(
        rhs,
        (z_acc, z_final),
        [np.log(ma)],
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
    )
    assert result.success
    return np.exp(result.y[0, -1])


def test_solver_default_is_picard_table_and_cached():
    solver = TidalStrippingSolver(M0=1e12, z_min=0.0, z_max=7.0)
    z_acc = 3.0
    ma = 1e-8 * solver.Mzvir(z_acc)

    default = solver.subhalo_mass_stripped(ma, z_acc, 0.0)
    explicit = solver.subhalo_mass_stripped(ma, z_acc, 0.0, method="picard_table")
    assert np.allclose(default, explicit, rtol=0, atol=0)
    assert len(solver._picard_tables) == 1

    table = solver._picard_tables[0.0]
    solver.subhalo_mass_stripped(10 * ma, z_acc, 0.0)
    assert solver._picard_tables[0.0] is table


def test_host_mass_change_invalidates_picard_cache():
    solver = TidalStrippingSolver(M0=1e12, z_min=0.0, z_max=7.0)
    z_acc = 1.0
    ma = 1e-6 * solver.Mzvir(z_acc)
    solver.subhalo_mass_stripped(ma, z_acc, 0.0)
    assert solver._picard_tables

    solver.M0 = 2e12
    assert solver._picard_tables == {}


def test_default_table_covers_cluster_microhalo_domain():
    solver = TidalStrippingSolver(M0=1e15, z_min=0.0, z_max=7.0)
    z_acc = 7.0
    ma = 1e-21 * solver.Mzvir(z_acc)
    reference = reference_log_ode_mass(solver, ma, z_acc)
    approximate = solver.subhalo_mass_stripped(ma, z_acc, 0.0)
    assert abs(approximate / reference - 1.0) < 1e-3


def test_high_level_public_defaults_use_picard_table():
    calc = inspect.signature(subhalo_properties.subhalo_properties_calc)
    obs = inspect.signature(subhalo_observables.__init__)
    assert calc.parameters["method"].default == "picard_table"
    assert obs.parameters["method"].default == "picard_table"


def test_notebook_documents_picard_validation():
    import json
    from pathlib import Path

    nb = json.loads(Path("approx_odeint.ipynb").read_text())
    tagged = [
        cell for cell in nb["cells"]
        if "picard-default-validation" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(tagged) >= 4
    joined = "\n".join("".join(cell.get("source", [])) for cell in tagged)
    assert "PicardTidalStrippingTable" in joined
    assert "three Picard iterations" in joined
    assert "picard_table" in joined
