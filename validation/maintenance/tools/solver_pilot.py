"""Frozen-domain Picard accuracy study, independent DOP853 integration."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import json
import hashlib
import platform
import time
import traceback

import numpy as np
from scipy.integrate import solve_ivp


def reference(solver,mass,za,times,refined=False):
    times=np.asarray(times)
    if times[-1]==za:
        return np.broadcast_to(mass,(len(times),len(mass))).copy()
    def rhs(z,y):
        return solver.Phi(z)*np.exp(solver.zetaMz(z)*(y-np.log(solver.Mzvir(z))))
    result=solve_ivp(rhs,(za,float(times[-1])),np.log(mass),t_eval=times,
                     method='DOP853',rtol=3e-13 if refined else 1e-11,
                     atol=3e-14 if refined else 1e-12)
    if not result.success or np.any(~np.isfinite(result.y)):
        raise RuntimeError(result.message)
    return np.exp(result.y.T)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    module=importlib.import_module('sashimi_'+args.variant)
    picard=importlib.import_module('picard_tidal_stripping')
    base=Path(module.__file__).parent
    gates=json.loads((base/'validation/maintenance/picard-gates.json').read_text())
    domain=gates['solver_domain']
    ratios=10**np.array(domain['log10_mass_ratios'])
    report={'variant':args.variant,'gates':gates['specification'],
            'python':platform.python_version(),
            'dependencies':{name:importlib.metadata.version(name) for name in ['numpy','scipy']},
            'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            'picard_sha256':hashlib.sha256(Path(picard.__file__).read_bytes()).hexdigest(),
            'cases':[],'failures':[],'tables':[]}
    begin=time.perf_counter()
    for host in domain['host_mass_msun']:
        try:
            if args.variant=='w':
                solver=module.TidalStrippingSolver(module.subhalos(2.),host)
            elif args.variant=='f':
                solver=module.TidalStrippingSolver(host,1.)
            else:
                solver=module.TidalStrippingSolver(host)
            for target in domain['observed_redshift']:
                tables={}
                if args.variant!='si':
                    for counts,iterations in [((96,128,128),3),((128,128,256),2),((128,128,256),3),((128,128,256),4),((192,256,512),3)]:
                        config=f'{counts[0]}x{counts[1]}x{counts[2]}-i{iterations}'
                        table=picard.PicardTidalStrippingTable(solver,z_final=target,
                            n_z_acc=counts[0],n_log_ratio=counts[1],n_integration=counts[2],n_iterations=iterations)
                        tables[config]=table
                        report['tables'].append({'host':host,'target':target,'config':config,
                            'seconds':table.build_seconds,'maximum_iteration_estimate':float(np.max(table.iteration_error))})
                for acc in sorted(set(min(target+offset,7.) for offset in domain['accretion_redshift_offsets'])):
                    mass=ratios*np.asarray(solver.Mzvir(acc)).item()
                    times=np.linspace(acc,target,100) if acc>target else np.array([target])
                    expected=reference(solver,mass,acc,times)
                    stricter=reference(solver,mass,acc,times,True)
                    residual=float(np.max(np.abs(expected/stricter-1)))
                    variants={}
                    if args.variant=='si':
                        for count in [100,129,257]:
                            for iterations in [2,3,4]:
                                config=f'history{count}-i{iterations}'
                                # Raw iteration results are assessed even when
                                # the runtime convergence guard would fall back.
                                if acc==target:
                                    actual=mass[None,:].copy()
                                else:
                                    grid,delta,estimate=picard._iterate(solver,np.log(mass),acc,target,count,iterations)
                                    t=(acc-times)/(acc-target)*(count-1)
                                    i=np.minimum(t.astype(int),count-2)
                                    f=t-i
                                    actual=mass[None,:]*np.exp(delta[i]*(1-f[:,None])+delta[i+1]*f[:,None])
                                variants[config]={'mass_relative_max':float(np.max(np.abs(actual/expected-1))),
                                                  'endpoint_mass':actual[-1].tolist()}
                    else:
                        for config,table in tables.items():
                            actual=table.mass(mass,acc)
                            variants[config]={'mass_relative_max':float(np.max(np.abs(actual/expected[-1]-1))),
                                              'endpoint_mass':np.asarray(actual).tolist(),
                                              'not_converged_count':int(np.sum(~table.converged(mass,acc)))}
                    report['cases'].append({'host':host,'observed_redshift':target,'accretion_redshift':acc,
                        'mass_ratio':ratios.tolist(),'reference_refinement_relative_max':residual,
                        'reference_endpoint_mass':expected[-1].tolist(),'candidates':variants})
                print(json.dumps({'host':host,'target':target,'cases':len(report['cases']),
                                  'seconds':time.perf_counter()-begin}),flush=True)
                args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        except Exception as exc:
            report['failures'].append({'host':host,'error':repr(exc),'traceback':traceback.format_exc()})
            print(json.dumps({'host':host,'failure':repr(exc)}),flush=True)
    report['elapsed_seconds']=time.perf_counter()-begin
    report['max_errors']={key:max(case['candidates'][key]['mass_relative_max'] for case in report['cases'])
                          for key in (report['cases'][0]['candidates'] if report['cases'] else [])}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'failures':len(report['failures']),'max_errors':report['max_errors']}),flush=True)


if __name__=='__main__':
    main()
