"""Validate the optimized candidate over the frozen physical domain."""
from pathlib import Path
import argparse,importlib,json,hashlib,traceback,warnings
import numpy as np
from independent_reference import reference


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--particle',type=float)
    args=parser.parse_args()
    module=importlib.import_module('sashimi_'+args.variant)
    helper=importlib.import_module('picard_tidal_stripping')
    base=Path(module.__file__).parent
    gates=json.loads((base/'validation/maintenance/picard-gates.json').read_text())
    ratios=10**np.linspace(-24.,3.,136)
    particle=args.particle if args.particle is not None else (2. if args.variant=='w' else 1.)
    report={'variant':args.variant,'particle':particle,'gates':gates['specification'],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(module.__file__),Path(helper.__file__)]},
        'cases':[],'domain_failures':[],
        'log10_mass_ratio_range':[-24.,3.],
        'domain_extension_reason':'The original frozen probe ended at +2. The full native W catalog reaches larger ratios at high accretion redshift; extend to +3 without changing error gates.'}
    for host in gates['solver_domain']['host_mass_msun']:
        try:
            if args.variant=='w':
                solver=module.TidalStrippingSolver(module.subhalos(particle),host)
            elif args.variant=='f':
                solver=module.TidalStrippingSolver(host,particle)
            else:
                solver=module.TidalStrippingSolver(host)
            for target in gates['solver_domain']['observed_redshift']:
                for acc in sorted(set(min(target+x,7.) for x in [0.,.001,.5,2.,5.])):
                    mass=ratios*np.asarray(solver.Mzvir(acc)).item()
                    times=np.linspace(acc,target,100) if acc>target else np.array([target])
                    expected=reference(solver,mass,acc,times)
                    refined=reference(solver,mass,acc,times,True)
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter('always')
                        if args.variant=='si':
                            actual=helper.history_mass(solver,mass,acc,times)
                            better=helper.history_mass(solver,mass,acc,times,n_iterations=4,n_integration=257,n_mass_nodes=128)
                        else:
                            actual=helper.endpoint_mass(solver,mass,acc,target)[None,:]
                            better=helper.endpoint_mass(solver,mass,acc,target,n_z_acc=96,n_log_ratio=64,n_integration=257,n_iterations=4)[None,:]
                            expected=expected[-1:];refined=refined[-1:]
                    row={'host':host,'target':target,'acc':acc,
                         'reference_refinement_max':float(np.max(np.abs(expected/refined-1))),
                         'mass_relative_max':float(np.max(np.abs(actual/expected-1))),
                         'refined_candidate_relative_max':float(np.max(np.abs(better/expected-1))),
                         'warnings':[str(w.message) for w in observed]}
                    report['cases'].append(row)
                args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(json.dumps({'host':host,'target':target,'max_error':max(r['mass_relative_max'] for r in report['cases'])}),flush=True)
        except Exception as exc:
            report['domain_failures'].append({'host':host,'error':repr(exc),'traceback':traceback.format_exc()})
            print(json.dumps({'host':host,'error':repr(exc)}),flush=True)
    report['maximum_mass_relative_error']=max((r['mass_relative_max'] for r in report['cases']),default=None)
    report['maximum_reference_refinement_error']=max((r['reference_refinement_max'] for r in report['cases']),default=None)
    report['fallbacks']=getattr(solver,'_picard_events',[])
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
