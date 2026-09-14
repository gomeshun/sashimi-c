"""Boundary, state identity, and analytic tests of the standalone solver."""
from pathlib import Path
import importlib
import warnings
import numpy as np
import pytest

from picard_tidal_stripping import (PicardTidalStrippingTable, endpoint_mass,
    history_mass, direct_log_mass, PicardFallbackWarning)


class AnalyticHost:
    def __init__(self, exponent=.01):
        self.M0=1e12
        self.z_min=0.
        self.z_max=7.
        self.exponent=exponent
        self.rate=.5
    def Mzvir(self,z):
        return np.ones_like(np.asarray(z))*self.M0
    def zetaMz(self,z):
        return np.ones_like(np.asarray(z))*self.exponent
    def Phi(self,z):
        return np.ones_like(np.asarray(z))*self.rate
    def exact(self,ma,za,z):
        if self.exponent==0:
            return ma*np.exp(self.rate*(z-za))
        return ma*(1-self.exponent*self.rate*(ma/self.M0)**self.exponent*(z-za))**(-1/self.exponent)


@pytest.mark.parametrize('exponent',[-.005,0.,.02])
def test_full_history_against_analytic_unexpanded_equation(exponent):
    solver=AnalyticHost(exponent)
    mass=np.array([1e-9,1e3,1e9])
    times=np.linspace(7.,0.,100)
    expected=solver.exact(mass[None,:],7.,times[:,None])
    actual=history_mass(solver,mass,7.,times,n_iterations=4,n_integration=257)
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=0)
    np.testing.assert_array_equal(actual[0],mass)
    assert not solver._picard_events


def test_shapes_no_evolution_and_partial_history_start():
    solver=AnalyticHost()
    assert isinstance(endpoint_mass(solver,1e6,0.,0.),float)
    np.testing.assert_array_equal(endpoint_mass(solver,[1.,2.],0.,0.),[1.,2.])
    assert endpoint_mass(solver,np.empty((0,2)),0.,0.).shape==(0,2)
    assert history_mass(solver,[],1.,[1.,.5,0.]).shape==(3,0)
    assert history_mass(solver,[1.,2.],1.,[]).shape==(0,2)
    assert history_mass(solver,np.ones((2,3)),1.,[1.,1.]).shape==(2,2,3)
    times=np.array([2.,1.,0.])
    actual=history_mass(solver,[1e6],3.,times,n_iterations=4,n_integration=257)
    np.testing.assert_allclose(actual[:,0],solver.exact(1e6,3.,times),rtol=1e-5)
    np.testing.assert_allclose(direct_log_mass(solver,[1e6],3.,times)[:,0],solver.exact(1e6,3.,times),rtol=1e-10)


@pytest.mark.parametrize('ma,za,z',[(0.,1.,0.),(-1.,1.,0.),(np.nan,1.,0.),
    (np.inf,1.,0.),(1.,np.nan,0.),(1.,1.,np.inf),(1.,0.,1.),(1.,0.,-1.)])
def test_rejects_nonphysical_solver_coordinates(ma,za,z):
    for method in [endpoint_mass,history_mass,direct_log_mass]:
        with pytest.raises(ValueError):
            method(AnalyticHost(),ma,za,z)


def test_history_direction_and_unknown_options_are_not_silently_ignored():
    solver=AnalyticHost()
    for method in [history_mass,direct_log_mass]:
        with pytest.raises(ValueError):
            method(solver,[1e6],3.,[3.,1.,2.])
    for method in [endpoint_mass,history_mass]:
        with pytest.raises(TypeError):
            method(solver,1e6,3.,0.,unknown_parameter=1)
        with pytest.raises(ValueError):
            method(solver,1e6,3.,0.,n_iterations=0)
        with pytest.raises(ValueError):
            method(solver,1e6,3.,0.,n_integration=1.5)


def test_table_cache_options_and_host_background_invalidation():
    solver=AnalyticHost()
    endpoint_mass(solver,[1e4,1e6],3.,0.)
    table=next(iter(solver._picard_tables.values()))
    endpoint_mass(solver,1e5,3.,0.)
    endpoint_mass(solver,1e5,3.,0.,n_iterations=4)
    solver.M0*=10
    with pytest.raises(ValueError,match='stale'):
        table.mass(1e5,3.)
    second=endpoint_mass(solver,[1e4,1e6],3.,0.)
    np.testing.assert_allclose(second, solver.exact(np.array([1e4,1e6]),3.,0.), rtol=1e-4)
    solver.rate=.7
    third=endpoint_mass(solver,[1e4,1e6],3.,0.)
    np.testing.assert_allclose(third, solver.exact(np.array([1e4,1e6]),3.,0.), rtol=1e-4)


def test_no_silent_extrapolation_and_counted_fallback():
    solver=AnalyticHost()
    table=PicardTidalStrippingTable(solver,log10_ratio_min=-12.,log10_ratio_max=-1.)
    with pytest.raises(ValueError):
        table.mass(1e-3,3.)
    with pytest.warns(PicardFallbackWarning):
        actual=endpoint_mass(solver,1e-15,3.,0.)
    np.testing.assert_allclose(actual,solver.exact(1e-15,3.,0.),rtol=1e-10)
    assert solver._picard_events[-1]['mass_count']==1
    with pytest.warns(PicardFallbackWarning):
        endpoint_mass(solver,1e6,7.,0.,n_iterations=1,convergence_tolerance=1e-15)
    assert 'convergence' in solver._picard_events[-1]['reason']


def test_public_dispatcher_candidate_and_legacy_paths():
    variant=next(v for v in ('c','si','w','f') if (Path(__file__).resolve().parents[1]/('sashimi_'+v+'.py')).exists())
    module=importlib.import_module('sashimi_'+variant)
    if variant=='w':
        solver=module.TidalStrippingSolver(module.subhalos(2.),1e12,z_max=3.)
    elif variant=='f':
        solver=module.TidalStrippingSolver(1e12,1.,z_max=3.)
    else:
        solver=module.TidalStrippingSolver(1e12,z_max=3.)
    method='picard' if variant=='si' else 'picard_table'
    mass=np.array([1e5,1e8])
    output=solver.subhalo_mass_stripped(mass,3.,0.,method=method)
    expected=direct_log_mass(solver,mass,3.,0.)
    np.testing.assert_allclose(np.asarray(output).ravel(),expected,rtol=1e-3)
    legacy=solver.subhalo_mass_stripped(mass,3.,0.,method='odeint',rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(np.asarray(legacy).ravel(),expected,rtol=2e-7)
    with pytest.raises(TypeError):
        solver.subhalo_mass_stripped(mass,3.,0.,method=method,mxstep=10)
    if variant!='w':
        with pytest.raises(TypeError):
            solver.subhalo_mass_stripped(mass,3.,0.,method='pert2_shanks',unused=1)


def test_long_solver_range_does_not_extend_an_in_domain_table():
    solver=AnalyticHost();solver.z_max=20.
    endpoint_mass(solver,1e6,3.,0.)
    assert next(iter(solver._picard_tables.values())).z_acc_max <= 7.+1e-13
    with pytest.warns(PicardFallbackWarning):
        endpoint_mass(solver,1e16,3.,0.)


def test_original_two_point_table_configuration_remains_supported():
    table=PicardTidalStrippingTable(AnalyticHost(),n_z_acc=2,n_log_ratio=2)
    assert np.isfinite(table.mass(1e6,3.))
    np.testing.assert_array_equal(table.mass([1.,2.],0.),[1.,2.])


def test_default_catalog_against_independent_ode_and_wrapper_forwarding(monkeypatch):
    import sashimi_c as module
    from ode_reference import reference

    model = module.subhalo_properties()
    options = dict(M0=1e12, dz=.5, zmax=1., N_ma=8, N_herm=2,
                   N_hermNa=2, logmamin=8., logmamax=10.)
    call = model.subhalo_properties_calc
    original = module.TidalStrippingSolver.subhalo_mass_stripped
    selected_methods = []

    def record(self, *args, **kwargs):
        selected_methods.append(kwargs.get('method'))
        return original(self, *args, **kwargs)

    def independent(self, mass, acc, times, **kwargs):
        values = reference(self, np.atleast_1d(mass), acc, np.atleast_1d(times), True)
        return values[-1] if np.ndim(times) == 0 else values

    for profile in [{'profile_change': True}, {'profile_change': False}]:
        with monkeypatch.context() as patch:
            patch.setattr(module.TidalStrippingSolver, 'subhalo_mass_stripped', record)
            with warnings.catch_warnings():
                warnings.simplefilter('error', PicardFallbackWarning)
                actual = call(**options, **profile)
        assert set(selected_methods) == {'picard_table'}
        with monkeypatch.context() as patch:
            patch.setattr(module.TidalStrippingSolver, 'subhalo_mass_stripped', independent)
            expected = call(**options, **profile, method='dop853')
        # The historical gates are 1e-3 for mass and 1% for smooth observables.
        # Individual continuous fields here must also agree within 0.5%.
        for i, (value, ref) in enumerate(zip(actual, expected)):
            assert np.all(np.isfinite(value))
            if i in (9,):
                np.testing.assert_array_equal(value, ref)
            else:
                np.testing.assert_allclose(value, ref, atol=0,
                    rtol=1e-3 if i == 4 else 5e-3, err_msg=f'column {i}')

    # Exercise positional/keyword forwarding without recalculating the catalog.
    from inspect import signature
    call_signature = signature(call)

    def catalog(self, *args, **kwargs):
        arguments = call_signature.bind(*args, **kwargs).arguments
        assert arguments['method'] == 'dop853'
        assert arguments['kwargs']['rtol'] == 2e-10
        assert arguments['profile_change'] is False
        return actual

    monkeypatch.setattr(module.subhalo_properties, 'subhalo_properties_calc', catalog)
    module.subhalo_observables(options['M0'], method='dop853',
        rtol=2e-10, profile_change=False).mass_function()
