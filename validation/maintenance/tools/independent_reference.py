"""Independent segmented log-mass DOP853 reference.

The Correa concentration fit changes branch at z=4. W's unchanged nearest
mass concentration table jumps at arithmetic midpoints of its mass nodes.
Segment those known coefficient boundaries so tolerance refinement cannot
miss a narrow jump. No Picard helper is imported.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def boundaries(solver,za,target):
    points=[float(za),float(target)]
    if not hasattr(solver,'model'):
        if target<4.<za: points.append(4.)
    else:
        import sashimi_w as w
        table=solver.model.filter_Mass
        mids=(table[1:]+table[:-1])/2
        def mass(z):
            m=solver.model.Mzzi(solver.M0,z,0.)
            if solver.N_hermNa==1 and solver.sigmafac!=0:
                m1=solver.model.Mzzi(solver.M0,1.,0.)
                scatter=(.12-.15*np.log10(m/solver.M0)) if z>1 else (.12-.15*np.log10(m1/solver.M0))/np.log10(m1/solver.M0)*np.log10(m/solver.M0)
                m=10**(np.log10(m)+solver.sigmafac*scatter)
                if solver.sigmafac>0: m=min(m,solver.M0)
            return m*w.h
        low,high=sorted([mass(za),mass(target)])
        for mid in mids[(mids>low)&(mids<high)]:
            points.append(brentq(lambda z:mass(z)-mid,target,za,xtol=1e-14))
    return sorted(set(points),reverse=True)


def reference(solver,mass,za,times,refined=False):
    times=np.asarray(times,dtype=float)
    mass=np.atleast_1d(mass)
    if times[-1]==za:
        return np.broadcast_to(mass,(len(times),len(mass))).copy()
    edges=boundaries(solver,za,float(times[-1]))
    log_initial=np.log(mass)
    # Integrate delta ln(m), so relative tolerances cannot depend on mass units.
    state=np.zeros_like(mass)
    out=np.empty((len(times),len(mass)))
    out[times==za]=log_initial
    for upper,lower in zip(edges[:-1],edges[1:]):
        def rhs(z,y):
            # Evaluate the appropriate one-sided value at a coefficient jump.
            z=np.clip(z,np.nextafter(lower,upper),np.nextafter(upper,lower))
            return solver.Phi(z)*np.exp(solver.zetaMz(z)*(y+log_initial-np.log(solver.Mzvir(z))))
        result=solve_ivp(rhs,(upper,lower),state,dense_output=True,
                         method='DOP853',rtol=3e-13 if refined else 1e-11,
                         atol=3e-14 if refined else 1e-12,
                         max_step=.05 if hasattr(solver,'model') else .2)
        if not result.success or np.any(~np.isfinite(result.y)):
            raise RuntimeError(result.message)
        selected=(times<upper)&(times>=lower)
        if np.any(selected):
            out[selected]=result.sol(times[selected]).T+log_initial
        state=result.y[:,-1]
    return np.exp(out)
