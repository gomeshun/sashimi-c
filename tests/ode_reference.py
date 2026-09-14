"""Independent segmented log-mass DOP853 reference for the native host.

The Correa concentration fit changes branch at z=4. Segment that boundary
so tolerance refinement cannot miss a narrow jump. No Picard helper is used.
"""
import numpy as np
from scipy.integrate import solve_ivp


def boundaries(solver, za, target):
    # The native Correa concentration fit changes branch at z=4.
    points = [float(za), float(target)]
    if target < 4. < za:
        points.append(4.)
    return sorted(set(points), reverse=True)


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
                         max_step=.2)
        if not result.success or np.any(~np.isfinite(result.y)):
            raise RuntimeError(result.message)
        selected=(times<upper)&(times>=lower)
        if np.any(selected):
            out[selected]=result.sol(times[selected]).T+log_initial
        state=result.y[:,-1]
    return np.exp(out)
