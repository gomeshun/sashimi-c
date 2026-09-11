"""Boundary, state identity, and analytic tests of the standalone solver."""
from pathlib import Path
import importlib
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
    assert solver._picard_tables=={}
    assert np.all(np.diff(actual,axis=0)<=0)


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
    first=endpoint_mass(solver,[1e4,1e6],3.,0.)
    table=next(iter(solver._picard_tables.values()))
    endpoint_mass(solver,1e5,3.,0.)
    assert len(solver._picard_tables)==1
    assert next(iter(solver._picard_tables.values())) is table
    endpoint_mass(solver,1e5,3.,0.,n_iterations=4)
    assert len(solver._picard_tables)==2
    solver.M0*=10
    with pytest.raises(ValueError,match='stale'):
        table.mass(1e5,3.)
    second=endpoint_mass(solver,[1e4,1e6],3.,0.)
    assert len(solver._picard_tables)==1
    assert not np.array_equal(first,second)
    solver.rate=.7
    third=endpoint_mass(solver,[1e4,1e6],3.,0.)
    assert len(solver._picard_tables)==1
    assert np.all(third<second)


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
    variant=next(v for v in ('c','si','w','f') if Path(__file__).with_name('sashimi_'+v+'.py').exists())
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


def test_public_default_matches_explicit_picard_and_observables():
    variant=next(v for v in ('c','si','w','f') if Path(__file__).with_name('sashimi_'+v+'.py').exists())
    module=importlib.import_module('sashimi_'+variant)
    cls={'c':'subhalo_properties','si':'subhalo_properties','w':'subhalos','f':'fdm_subhalo_properties'}[variant]
    model=getattr(module,cls)()
    options=dict(M0=1e12,dz=.5,zmax=1.,N_ma=8,N_herm=2,N_hermNa=2,logmamin=8.,logmamax=10.)
    method='picard' if variant=='si' else 'picard_table'
    call=model.rs_rhos_calc if variant=='w' else model.subhalo_properties_calc
    for profile_change in [True,False] if variant!='si' else [None]:
        profile={} if profile_change is None else {'profile_change':profile_change}
        actual=call(**options,**profile)
        expected=call(**options,**profile,method=method)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    if variant=='w':
        for name,args in [('subhalo_distr',()),('N_sat',()),('N_sat_Vthres',(10.,))]:
            observable=getattr(model,name)
            opts={k:v for k,v in options.items() if k!='M0'}
            a=observable(options['M0'],*args,**opts)
            b=observable(options['M0'],*args,**opts,method=method)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    elif variant in ['c','f']:
        cls=getattr(module,'subhalo_observables' if variant=='c' else 'fdm_subhalo_observables')
        opts={k:v for k,v in options.items() if k!='M0'}
        a=cls(M0_per_Msun=options['M0'],**opts)
        b=cls(M0_per_Msun=options['M0'],**opts,method=method)
        np.testing.assert_array_equal(a.m0,b.m0)
        for x,y in zip(a.mass_function(),b.mass_function()):np.testing.assert_array_equal(x,y)



def test_original_two_point_table_configuration_remains_supported():
    table=PicardTidalStrippingTable(AnalyticHost(),n_z_acc=2,n_log_ratio=2)
    assert table.interpolation=='linear'
    assert np.isfinite(table.mass(1e6,3.))
    np.testing.assert_array_equal(table.mass([1.,2.],0.),[1.,2.])
