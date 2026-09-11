"""Standalone Picard tidal mass loss, derived from SASHIMI-C PR #5.

Origin: gomeshun/sashimi-c 88ae730762fb153be7a7433bb563b0b8ab3ec2c2.
Only the numerical integral equation is shared. Each caller supplies its own
Phi, zetaMz, and Mzvir, in its existing units. No variant or ITAMAE import.

The unexpanded nonlinear factor is exp(zeta * log(m/Mvir)). Endpoint tables
and the full-history integrator use the same Picard updates. The default
candidate has three updates; one further update estimates convergence.
"""
# MIT License
#
# Copyright (c) 2026 Shin'ichiro Ando
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline


class PicardFallbackWarning(UserWarning):
    """An explicitly counted high-accuracy ODE fallback was required."""


def _real(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or np.any(~np.isfinite(a)):
        raise ValueError(f'{name} must contain finite real numbers.')
    return a.astype(float, copy=False)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def physics_key(solver):
    """Track numerical host/background inputs, including changed power arrays.

    Solver-owned caches are excluded. Custom scalar history settings and
    instance-level callable replacements participate in the key as well.
    """
    result = []
    for name, value in sorted(vars(solver).items()):
        if name.startswith(('_picard', '_eps', 'eps_')):
            continue
        if isinstance(value, (float, int, str, bool, np.number)):
            result.append((name, repr(value)))
        elif isinstance(value, np.ndarray) and value.dtype.kind in 'iuf':
            result.append((name, value.shape, hashlib.sha256(value.tobytes()).hexdigest()))
        elif callable(value):
            result.append((name, id(value)))
    if hasattr(solver, 'picard_physics_key'):
        result.append(('provider', solver.picard_physics_key()))
    for name in ('Phi', 'zetaMz', 'Mzvir'):
        value = getattr(solver, name)
        result.append((name, id(getattr(value, '__func__', value))))
    return tuple(result)


def _state(solver):
    key = physics_key(solver)
    if getattr(solver, '_picard_physics_key', None) != key:
        solver._picard_tables = {}
        solver._picard_coefficients = {}
        solver._picard_host_cache = {}
        solver._picard_scalar_cache = {}
        solver._picard_physics_key = key
    if not hasattr(solver, '_picard_events'):
        solver._picard_events = []
    return key


def cached_host_mass(solver, z):
    """Memoize the unchanged host conversion, including scalar ODE stages."""
    _state(solver)
    redshift = np.asarray(z, dtype=float)
    key = (redshift.shape, redshift.tobytes())
    cache = solver._picard_host_cache
    if key not in cache:
        value = solver._Mzvir_uncached(z)
        if len(cache)>=4096:
            cache.pop(next(iter(cache)))
        cache[key] = np.asarray(value).copy()
    result = cache[key].copy()
    return float(result) if result.ndim==0 else result


def cached_scalar_variance(solver, mass, calculate):
    """Cache repeated scalar native integrals on a fixed host/power identity."""
    if np.ndim(mass)!=0:
        return calculate(mass)
    _state(solver)
    key = float(mass)
    cache = solver._picard_scalar_cache
    if key not in cache:
        if len(cache)>=256:
            cache.pop(next(iter(cache)))
        cache[key] = calculate(mass)
    return cache[key]


def _coefficients(solver, z_acc, z_final, count):
    _state(solver)
    key = (float(z_acc), float(z_final), count)
    cached = solver._picard_coefficients.get(key)
    if cached is not None:
        return cached
    z = np.linspace(z_acc, z_final, count)
    phi = np.broadcast_to(np.asarray(solver.Phi(z), dtype=float), z.shape)
    zeta = np.broadcast_to(np.asarray(solver.zetaMz(z), dtype=float), z.shape)
    host = np.broadcast_to(np.asarray(solver.Mzvir(z), dtype=float), z.shape)
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(zeta)) or not np.all(np.isfinite(host)) or np.any(host <= 0) or np.any(phi < 0):
        raise ValueError('Tidal stripping requires finite, positive host mass and nonnegative rate.')
    result = z, phi, zeta, np.log(host)
    # Bound retained coefficient paths; SI catalogs may contain many slices.
    if len(solver._picard_coefficients) >= 512:
        solver._picard_coefficients.pop(next(iter(solver._picard_coefficients)))
    solver._picard_coefficients[key] = result
    return result


def _iterate(solver, log_ma, z_acc, z_final, n_integration, n_iterations, coefficients=None):
    z, phi, zeta, log_host = (_coefficients(solver, z_acc, z_final, n_integration)
                              if coefficients is None else coefficients)
    log_ma = np.asarray(log_ma)
    initial = cumulative_trapezoid(phi, x=z, initial=0.)
    delta = np.broadcast_to(initial[:, None], (len(z), log_ma.size)).copy()
    log_ratio = log_ma[None, :] - log_host[:, None]

    def update(current):
        rhs = phi[:, None]*np.exp(zeta[:, None]*(log_ratio+current))
        return cumulative_trapezoid(rhs, x=z, axis=0, initial=0.)

    for _ in range(n_iterations):
        delta = update(delta)
    next_delta = update(delta)
    residual = np.max(np.abs(next_delta-delta), axis=0)
    # A conservative local contraction estimate from the integrated rate.
    contraction = np.max(np.abs(zeta))*np.max(np.abs(next_delta), axis=0)
    estimate = np.full(residual.shape, np.inf)
    valid = contraction < 1.
    estimate[valid] = residual[valid]/(1-contraction[valid])
    return z, delta, estimate


def direct_log_mass(solver, ma, za, z, *, rtol=1e-11, atol=1e-12):
    """DOP853 integration of log mass, preserving scalar or full-history shape."""
    mass = _real(ma, 'ma')
    redshifts = _real(z, 'z')
    za = float(_real(za, 'za'))
    if np.any(mass <= 0) or za <= -1 or np.any(redshifts <= -1) or np.any(redshifts > za):
        raise ValueError('Require ma>0 and -1<z<=za.')
    scalar_z = redshifts.ndim == 0
    times = np.atleast_1d(redshifts)
    if times.ndim != 1 or np.any(np.diff(times) > 0):
        raise ValueError('History redshifts must be one-dimensional and non-increasing.')
    if mass.size == 0 or times.size == 0:
        return np.empty(mass.shape if scalar_z else times.shape+mass.shape)
    if times[-1] == za:
        return mass.copy() if scalar_z else np.broadcast_to(mass,times.shape+mass.shape).copy()
    def rhs(redshift, log_mass):
        return solver.Phi(redshift)*np.exp(solver.zetaMz(redshift)*(log_mass-np.log(solver.Mzvir(redshift))))
    # Integrate from the declared accretion time even if it is not in t_eval.
    unique, reverse = np.unique(times, return_inverse=True)
    result = solve_ivp(rhs, (za,float(times[-1])), np.log(mass.ravel()),
                       t_eval=unique[::-1], method='DOP853', rtol=rtol, atol=atol)
    if not result.success or np.any(~np.isfinite(result.y)):
        raise RuntimeError('Log-mass ODE failed: '+result.message)
    answer = np.exp(result.y.T[::-1][reverse]).reshape(times.shape+mass.shape)
    answer = answer[0] if scalar_z else answer
    return float(answer) if answer.ndim == 0 else answer


def _fallback(solver, ma, za, z, reason):
    _state(solver)
    event = dict(reason=reason, mass_count=int(np.size(ma)), z_acc=float(za),
                 z_final=float(np.min(np.atleast_1d(z))))
    solver._picard_events.append(event)
    warnings.warn(f'Picard fallback to log-mass DOP853: {reason}; {event["mass_count"]} masses.',
                  PicardFallbackWarning, stacklevel=3)
    return direct_log_mass(solver, ma, za, z)


class PicardTidalStrippingTable:
    """Picard endpoint table; mass and redshift queries never extrapolate.

    The convergence estimate covers the nonlinear iterations. Interpolation
    and quadrature errors are separately validated by the saved ODE campaign.
    """
    def __init__(self, solver, z_final=None, z_acc_max=None, n_z_acc=48,
                 n_log_ratio=32, log10_ratio_min=-24., log10_ratio_max=3.,
                 n_integration=129, n_iterations=3, convergence_tolerance=8e-4,
                 interpolation='cubic'):
        self.solver = solver
        self.z_final = float(_real(solver.z_min if z_final is None else z_final,'z_final'))
        self.z_acc_max = float(_real(solver.z_max if z_acc_max is None else z_acc_max,'z_acc_max'))
        if self.z_final <= -1 or self.z_acc_max < self.z_final:
            raise ValueError('Require -1 < z_final <= z_acc_max.')
        self.n_z_acc = _integer(n_z_acc,'n_z_acc',2)
        self.n_log_ratio = _integer(n_log_ratio,'n_log_ratio',2)
        self.n_integration = _integer(n_integration,'n_integration',2)
        self.n_iterations = _integer(n_iterations,'n_iterations',1)
        self.log10_ratio_min = float(_real(log10_ratio_min,'log10_ratio_min'))
        self.log10_ratio_max = float(_real(log10_ratio_max,'log10_ratio_max'))
        if self.log10_ratio_max <= self.log10_ratio_min:
            raise ValueError('log10_ratio_max must exceed log10_ratio_min.')
        self.convergence_tolerance = float(_real(convergence_tolerance,'convergence_tolerance'))
        if self.convergence_tolerance <= 0:
            raise ValueError('convergence_tolerance must be positive.')
        if interpolation not in ('linear','cubic'):
            raise ValueError('interpolation must be linear or cubic.')
        if interpolation=='cubic' and min(self.n_z_acc,self.n_log_ratio)<4:
            raise ValueError('Cubic tables need at least four nodes on each axis.')
        self.interpolation = interpolation
        self.z_acc_grid = np.linspace(self.z_final,self.z_acc_max,self.n_z_acc)
        self.log_ratio_grid = np.linspace(self.log10_ratio_min,self.log10_ratio_max,self.n_log_ratio)
        self._ln_ratio_grid = self.log_ratio_grid*np.log(10.)
        self.delta_ln_mass = np.zeros((self.n_z_acc,self.n_log_ratio))
        self.iteration_error = np.zeros_like(self.delta_ln_mass)
        self._physics_key = _state(solver)
        start = time.perf_counter()
        # Evaluate each host/background array once for the complete table.
        # Flattening retains the one-dimensional provider interface.
        paths = np.linspace(self.z_acc_grid,self.z_final,self.n_integration,axis=-1)
        flat = paths.ravel()
        phi = np.broadcast_to(np.asarray(solver.Phi(flat)),flat.shape).reshape(paths.shape)
        zeta = np.broadcast_to(np.asarray(solver.zetaMz(flat)),flat.shape).reshape(paths.shape)
        host = np.broadcast_to(np.asarray(solver.Mzvir(flat)),flat.shape).reshape(paths.shape)
        if np.any(~np.isfinite(host)) or np.any(host<=0) or np.any(~np.isfinite(phi)) or np.any(phi<0) or np.any(~np.isfinite(zeta)):
            raise ValueError('Tidal table requires finite, positive host mass and nonnegative rate.')
        log_host = np.log(host)
        for i, za in enumerate(self.z_acc_grid):
            if za == self.z_final:
                continue
            log_ma = log_host[i,0]+self._ln_ratio_grid
            _, history, error = _iterate(solver,log_ma,za,self.z_final,self.n_integration,self.n_iterations,
                                          (paths[i],phi[i],zeta[i],log_host[i]))
            self.delta_ln_mass[i] = history[-1]
            self.iteration_error[i] = error
        self.build_seconds = time.perf_counter()-start
        if self.z_final == self.z_acc_max:
            self._interpolator = None
        elif interpolation=='linear':
            self._interpolator = RegularGridInterpolator(
                (self.z_acc_grid,self._ln_ratio_grid),self.delta_ln_mass,bounds_error=True)
        else:
            self._interpolator = RectBivariateSpline(self.z_acc_grid,self._ln_ratio_grid,
                                                     self.delta_ln_mass,kx=3,ky=3,s=0)

    def _points(self, ma, z_acc):
        if physics_key(self.solver) != self._physics_key:
            raise ValueError('Picard table is stale after a host or background change.')
        mass, za = np.broadcast_arrays(_real(ma,'ma'),_real(z_acc,'z_acc'))
        if np.any(mass <= 0):
            raise ValueError('Accretion masses must be positive.')
        if np.any(za<self.z_final) or np.any(za>self.z_acc_max):
            raise ValueError('z_acc lies outside the precomputed redshift range.')
        if mass.size == 0:
            return mass,np.empty((0,2))
        host = np.asarray(self.solver.Mzvir(za))
        ratio = np.log(mass/host)
        if np.any(~np.isfinite(ratio)) or np.any(ratio<self._ln_ratio_grid[0]) or np.any(ratio>self._ln_ratio_grid[-1]):
            raise ValueError('ma/Mvir(z_acc) lies outside the precomputed mass-ratio range.')
        return mass,np.column_stack((za.ravel(),ratio.ravel()))

    def delta_log_mass(self, ma, z_acc):
        mass, points = self._points(ma,z_acc)
        if self._interpolator is None or mass.size == 0:
            values = np.zeros_like(mass)
        else:
            if self.interpolation=='linear':
                values = self._interpolator(points).reshape(mass.shape)
            else:
                values = self._interpolator.ev(points[:,0],points[:,1]).reshape(mass.shape)
        return float(values) if values.ndim == 0 else values

    def mass(self, ma, z_acc):
        mass = np.asarray(ma)
        values = mass*np.exp(self.delta_log_mass(ma,z_acc))
        return float(values) if values.ndim == 0 else values

    def converged(self, ma, z_acc):
        mass, points = self._points(ma,z_acc)
        if mass.size == 0 or self._interpolator is None:
            return np.ones(mass.shape,dtype=bool)
        i = np.clip(np.searchsorted(self.z_acc_grid,points[:,0])-1,0,self.n_z_acc-2)
        j = np.clip(np.searchsorted(self._ln_ratio_grid,points[:,1])-1,0,self.n_log_ratio-2)
        offsets = (-1,0,1,2) if self.interpolation=='cubic' else (0,1)
        error = np.maximum.reduce([self.iteration_error[np.clip(i+di,0,self.n_z_acc-1),
                                                        np.clip(j+dj,0,self.n_log_ratio-1)]
                                   for di in offsets for dj in offsets])
        return (error<=self.convergence_tolerance).reshape(mass.shape)


def endpoint_mass(solver, ma, za, z, *, n_z_acc=48, n_log_ratio=32,
                  log10_ratio_min=-24., log10_ratio_max=3., n_integration=129,
                  n_iterations=3, convergence_tolerance=8e-4, interpolation='cubic'):
    n_z_acc = _integer(n_z_acc,'n_z_acc',2)
    n_log_ratio = _integer(n_log_ratio,'n_log_ratio',2)
    n_integration = _integer(n_integration,'n_integration',2)
    n_iterations = _integer(n_iterations,'n_iterations',1)
    log10_ratio_min = float(_real(log10_ratio_min,'log10_ratio_min'))
    log10_ratio_max = float(_real(log10_ratio_max,'log10_ratio_max'))
    if log10_ratio_max <= log10_ratio_min:
        raise ValueError('log10_ratio_max must exceed log10_ratio_min.')
    convergence_tolerance = float(_real(convergence_tolerance,'convergence_tolerance'))
    if convergence_tolerance <= 0:
        raise ValueError('convergence_tolerance must be positive.')
    if interpolation not in ('linear','cubic'):
        raise ValueError('interpolation must be linear or cubic.')
    if interpolation=='cubic' and min(n_z_acc,n_log_ratio)<4:
        raise ValueError('Cubic tables need at least four nodes on each axis.')
    mass, acc, final = np.broadcast_arrays(_real(ma,'ma'),_real(za,'za'),_real(z,'z'))
    if np.any(mass<=0) or np.any(acc<=-1) or np.any(final<=-1) or np.any(final>acc):
        raise ValueError('Require ma>0 and -1<z<=za.')
    _state(solver)
    output = mass.copy()
    if mass.size == 0:
        return output
    # Group distinct final redshifts while preserving the usual scalar-target
    # table reuse. SI uses history_mass instead of creating endpoint tables.
    for target in np.unique(final):
        selected = (final==target) & (acc>target)
        if not np.any(selected):
            continue
        m, a = mass[selected], acc[selected]
        host = np.asarray(solver.Mzvir(a))
        if np.any(~np.isfinite(host)) or np.any(host <= 0):
            raise ValueError('Tidal stripping requires finite positive host mass.')
        ratio = np.log10(m/host)
        outside = (a>7.+32*np.finfo(float).eps) | (target<0.) | (ratio<-24.) | (ratio>2.+32*np.finfo(float).eps)
        selected_output = np.empty_like(m)
        for start in np.unique(a[outside]):
            use = outside & (a==start)
            selected_output[use] = _fallback(solver,m[use],start,target,'outside validated redshift or mass-ratio domain')
        inside = ~outside
        if np.any(inside):
            # Required coordinates are included even when custom table limits
            # are narrower. Expansion is explicit in the cached table metadata.
            lo = min(float(log10_ratio_min),float(np.floor(np.min(ratio[inside]))))-1e-10
            hi = max(float(log10_ratio_max),float(np.ceil(np.max(ratio[inside]))))+1e-10
            zmax = max(min(float(solver.z_max),7.)+32*np.finfo(float).eps,float(np.max(a[inside])))
            key = (float(target),zmax,n_z_acc,n_log_ratio,lo,hi,n_integration,n_iterations,convergence_tolerance,interpolation)
            table = solver._picard_tables.get(key)
            if table is None:
                table = PicardTidalStrippingTable(solver,target,zmax,n_z_acc,n_log_ratio,lo,hi,
                    n_integration,n_iterations,convergence_tolerance,interpolation)
                solver._picard_tables[key] = table
            ok = table.converged(m[inside],a[inside])
            values = table.mass(m[inside],a[inside])
            for start in np.unique(a[inside][~ok]):
                use = (~ok) & (a[inside]==start)
                values[use] = _fallback(solver,m[inside][use],start,target,'Picard iteration convergence estimate exceeded tolerance')
            selected_output[inside] = values
        output[selected] = selected_output
    return float(output) if output.ndim == 0 else output


def history_mass(solver, ma, za, z, *, n_integration=100, n_iterations=3,
                 convergence_tolerance=8e-4, n_mass_nodes=32):
    mass = _real(ma,'ma')
    times = _real(z,'z')
    za = float(_real(za,'za'))
    scalar_z = times.ndim == 0
    times = np.atleast_1d(times)
    count = _integer(n_integration,'n_integration',2)
    iterations = _integer(n_iterations,'n_iterations',1)
    mass_nodes = _integer(n_mass_nodes,'n_mass_nodes',2)
    if np.any(mass<=0) or za<=-1 or np.any(times<=-1) or np.any(times>za):
        raise ValueError('Require ma>0 and -1<z<=za.')
    if times.ndim != 1 or np.any(np.diff(times)>0):
        raise ValueError('History redshifts must be one-dimensional and non-increasing.')
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0:
        raise ValueError('convergence_tolerance must be finite and positive.')
    _state(solver)
    if mass.size == 0 or times.size == 0:
        return np.empty(mass.shape if scalar_z else times.shape+mass.shape)
    if times[-1] == za:
        return mass.copy() if scalar_z else np.broadcast_to(mass,times.shape+mass.shape).copy()
    ratio = np.log10(mass/solver.Mzvir(za))
    if za>7.+32*np.finfo(float).eps or times[-1]<0 or np.any(ratio<-24) or np.any(ratio>2.+32*np.finfo(float).eps):
        return _fallback(solver,mass,za,z,'outside validated redshift or mass-ratio domain')
    log_mass = np.log(mass.ravel())
    # Wider requests receive enough mass nodes to keep the interpolation
    # spacing within the validated quarter-dex bound.
    mass_nodes = max(mass_nodes,int(np.ceil(np.ptp(log_mass)/(np.log(10.)*.25)))+1)
    use_mass_grid = log_mass.size>mass_nodes and np.ptp(log_mass)>0
    nodes = np.linspace(np.min(log_mass),np.max(log_mass),mass_nodes) if use_mass_grid else log_mass
    grid, delta, error = _iterate(solver,nodes,za,float(times[-1]),count,iterations)
    # Vectorized linear interpolation of the entire requested trajectory.
    if np.array_equal(grid,times):
        requested = delta
    else:
        positions = (za-times)/(za-times[-1])*(count-1)
        left = np.minimum(positions.astype(int),count-2)
        fraction = positions-left
        requested = delta[left]*(1-fraction[:,None])+delta[left+1]*fraction[:,None]
    if use_mass_grid:
        coordinate = (log_mass-nodes[0])/(nodes[-1]-nodes[0])*(mass_nodes-1)
        left = np.minimum(coordinate.astype(int),mass_nodes-2)
        fraction = coordinate-left
        requested = requested[:,left]*(1-fraction[None,:])+requested[:,left+1]*fraction[None,:]
        error = np.maximum(error[left],error[left+1])
    output = (mass.ravel()[None,:]*np.exp(requested)).reshape(times.shape+mass.shape)
    bad = error>convergence_tolerance
    if np.any(bad):
        flattened = output.reshape(len(times),-1)
        flattened[:,bad] = _fallback(solver,mass.ravel()[bad],za,times,
            'Picard iteration convergence estimate exceeded tolerance')
    output = output[0] if scalar_z else output
    return float(output) if output.ndim == 0 else output
