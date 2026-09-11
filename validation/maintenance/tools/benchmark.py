"""Process-isolated benchmark; run variants sequentially on an idle host."""
import argparse, hashlib, importlib, json, platform, resource, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import scipy


def worker(variant, mode):
    module=importlib.import_module('sashimi_'+variant)
    base=Path(module.__file__).parent
    gates=json.loads((base/'validation/maintenance/picard-gates.json').read_text())
    config=gates['catalogs'][variant]
    cls={'c':'subhalo_properties','si':'subhalo_properties','w':'subhalos','f':'fdm_subhalo_properties'}[variant]
    method=('odeint' if variant=='w' else 'pert2_shanks') if mode=='legacy' else ('picard' if variant=='si' else 'picard_table')
    start=time.perf_counter()
    model=getattr(module,cls)()
    call=model.rs_rhos_calc if variant=='w' else model.subhalo_properties_calc
    arrays=call(**config,method=method)
    elapsed=time.perf_counter()-start
    assert all(np.all(np.isfinite(a)) for a in arrays)
    solver=model.stripping_solver
    result={'whole_catalog_seconds':elapsed, 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'catalog_table_build_seconds':sum(t.build_seconds for t in getattr(solver,'_picard_tables',{}).values()),
       'catalog_table_count':len(getattr(solver,'_picard_tables',{})),
       'catalog_fallback_events':getattr(solver,'_picard_events',[])}
    if mode=='picard':
        if variant=='w': fresh=module.TidalStrippingSolver(model,1e12,z_max=7.)
        elif variant=='f': fresh=module.TidalStrippingSolver(1e12,1.,z_max=7.)
        else: fresh=module.TidalStrippingSolver(1e12,z_max=7.)
        masses=np.geomspace(1e-6 if variant=='c' else 1e6,1e11,128)
        times=np.linspace(7.,0.,100) if variant=='si' else 0.
        start=time.perf_counter(); first=fresh.subhalo_mass_stripped(masses,7.,times,method=method); first_time=time.perf_counter()-start
        start=time.perf_counter(); reused=fresh.subhalo_mass_stripped(masses,7.,times,method=method); reuse_time=time.perf_counter()-start
        np.testing.assert_array_equal(first,reused)
        result.update(first_solver_seconds=first_time,reused_solver_seconds=reuse_time,
            solver_table_build_seconds=sum(t.build_seconds for t in fresh._picard_tables.values()),
            solver_fallback_events=fresh._picard_events)
    print(json.dumps(result,allow_nan=False))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',required=True); p.add_argument('--worker',choices=['legacy','picard']); p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.worker: return worker(args.variant,args.worker)
    report={'variant':args.variant,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
       'platform':platform.platform(),'repetitions':3,'runs':{'legacy':[],'picard':[]},
       'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('.').glob('*.py') if not f.name.startswith('test')},
       'measurement':'Each mode and repetition is a fresh process; whole catalog includes constructor and table build. Peak RSS is sampled before solver microbenchmark. No competing validation jobs.'}
    for i in range(3):
        for mode in ['legacy','picard']:
            command=[sys.executable,__file__,'--variant',args.variant,'--worker',mode]
            child=subprocess.run(command,text=True,capture_output=True,check=True)
            data=json.loads(child.stdout.strip().splitlines()[-1]); data['stderr']=child.stderr
            report['runs'][mode].append(data)
            print(args.variant,i,mode,data['whole_catalog_seconds'],flush=True)
    med={mode:statistics.median(x['whole_catalog_seconds'] for x in rows) for mode,rows in report['runs'].items()}
    report['median_seconds']=med; report['whole_catalog_speedup']=med['legacy']/med['picard']
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.variant,'speedup',report['whole_catalog_speedup'],flush=True)

if __name__=='__main__': main()
