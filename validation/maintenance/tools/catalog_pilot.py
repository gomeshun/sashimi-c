"""Same-catalog solver comparison with fixed observables and saved arrays."""
from pathlib import Path
import argparse
import importlib
import json
import hashlib
import time
import traceback
import warnings
import resource

import numpy as np
from scipy.integrate import solve_ivp


from independent_reference import reference


def independent_solve(self,mass,acc,times,method=None,**kwargs):
    scalar=np.ndim(times)==0
    result=reference(self,np.atleast_1d(mass),acc,np.atleast_1d(times),True)
    return result[-1] if scalar else result


def metrics(variant,module,model,arrays,config):
    if variant=='si':
        ma,z=arrays[:2]; mass=arrays[11]; radius=arrays[16]; density=arrays[17]
        weights=arrays[24]; survival=arrays[26].astype(bool)
        vmax=arrays[20]/(model.km/model.s); vpeak=arrays[10]/(model.km/model.s)
        core=arrays[18]/model.kpc
    else:
        ma,z,ra,rhoa,mass,radius,density,ct,weights,survival=arrays
        if variant=='w':
            radius=radius*module.kpc; density=density*(module.Msolar/module.pc**3)
            ra=ra*module.kpc; rhoa=rhoa*(module.Msolar/module.pc**3)
            gravity=module.G; velocity_unit=module.km/module.s
        else:
            gravity=model.G; velocity_unit=model.km/model.s
        vmax=np.sqrt(4*np.pi*gravity*density/4.625)*radius/velocity_unit
        vpeak=np.sqrt(4*np.pi*gravity*rhoa/4.625)*ra/velocity_unit
    selected=np.asarray(survival,dtype=bool)
    w=weights*selected
    total=float(np.sum(w))
    bins=np.geomspace(1e-10,.2*config['M0'],101)
    histogram=np.histogram(mass,bins=bins,weights=w)[0]
    output={'survivor_count':total,'satellites_mpeak_gt_1e8':float(np.sum(w[ma>1e8])),
            'satellites_vpeak_gt_10':float(np.sum(w[vpeak>10])),
            'bound_mass_fraction':float(np.sum(w*mass)/config['M0']),
            'mean_vmax_km_s':float(np.sum(w*vmax)/total) if total else 0.,
            'mass_function_bins_msun':bins.tolist(),'mass_function_counts':histogram.tolist()}
    if variant=='si':
        output['mean_core_radius_kpc']=float(np.sum(w*core)/total) if total else 0.
        output['collapse_weight_fraction']=float(np.sum(w[arrays[22]>=model.param_model.tt_th])/total) if total else 0.
        output['cdm_survivor_count']=float(np.sum(arrays[23]))
    elif variant in ('c','f'):
        cls=module.subhalo_observables if variant=='c' else module.fdm_subhalo_observables
        obs=cls.__new__(cls); obs.__dict__.update(model.__dict__)
        for name,value in zip(('ma200','z_a','rs_a','rhos_a','m0','rs0','rhos0','ct0','weight'),arrays[:9]):
            setattr(obs,name,value[selected])
        obs.Vmax=vmax[selected]*(model.km/model.s)
        obs.Vpeak=vpeak[selected]*(model.km/model.s)
        obs.rmax=2.163*obs.rs0; obs.rpeak=2.163*obs.rs_a
        output['public_bound_mass_fraction']=float(obs.mass_fraction())
        output['boost_n0']=float(obs.annihilation_boost_factor(n=0)[0])
    else:
        original=model.rs_rhos_calc
        model.rs_rhos_calc=lambda *a,**k: arrays
        try:
            output['public_satellites_mpeak_gt_1e8']=model.N_sat(config['M0'],Mpeak=1e8,Mpeak_thres=True)[0]
            output['public_satellites_vpeak_gt_10']=model.N_sat_Vthres(config['M0'],10.)[0]
        finally:
            model.rs_rhos_calc=original
    return output


def compare_runs(report,directory):
    if 'picard' in report['runs'] and 'reference' in report['runs']:
        a=np.load(directory/report['runs']['picard']['array_file'])
        b=np.load(directory/report['runs']['reference']['array_file'])
        mi,wi,masks=(11,23,[25,26]) if report['variant']=='si' else (4,8,[9])
        masserr=np.abs(a[f'tuple_{mi}']/b[f'tuple_{mi}']-1)
        report['comparison']={'maximum_mass_relative_error':float(np.max(masserr)),
            'changed_masks':{},'metric_differences':{}}
        for i in masks:
            changed=a[f'tuple_{i}']!=b[f'tuple_{i}']
            weights=np.maximum(a[f'tuple_{wi}'],b[f'tuple_{wi}'])
            report['comparison']['changed_masks'][str(i)]={'weight':float(np.sum(weights[changed])),
                'weight_fraction':float(np.sum(weights[changed])/np.sum(weights)),
                'nodes':[{'index':int(j),'ma_msun':float(b['tuple_0'][j]),'z_acc':float(b['tuple_1'][j]),
                          'reference':bool(b[f'tuple_{i}'][j]),'picard':bool(a[f'tuple_{i}'][j]),
                          'weight':float(weights[j])} for j in np.flatnonzero(changed)]}
        p=report['runs']['picard']['metrics']; r=report['runs']['reference']['metrics']
        for name,value in p.items():
            if isinstance(value,(int,float)):
                difference=abs(value-r[name])
                report['comparison']['metric_differences'][name]={'absolute':difference,
                    'relative':difference/abs(r[name]) if r[name] else None}
        report['comparison']['mass_function_l1_fraction']=float(np.sum(np.abs(np.array(p['mass_function_counts'])-r['mass_function_counts']))/r['survivor_count'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--modes',default='legacy,picard,reference')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--picard-options',type=Path)
    args=parser.parse_args()
    module=importlib.import_module('sashimi_'+args.variant)
    base=Path(module.__file__).parent
    gates=json.loads((base/'validation/maintenance/picard-gates.json').read_text())
    config=dict(gates['catalogs'][args.variant])
    constructor={}
    if args.config:
        inputs=json.loads(args.config.read_text())
        config.update(inputs.get('parameters',{})); constructor=inputs.get('constructor',{})
    picard_options=json.loads(args.picard_options.read_text()) if args.picard_options else {}
    cls={'c':'subhalo_properties','si':'subhalo_properties','w':'subhalos','f':'fdm_subhalo_properties'}[args.variant]
    report={'variant':args.variant,'config':config,'constructor':constructor,'picard_options':picard_options,
            'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            'gates':gates['specification'],'runs':{},'failures':[]}
    original=module.TidalStrippingSolver.subhalo_mass_stripped
    for mode in args.modes.split(','):
        start=time.perf_counter()
        try:
            model=getattr(module,cls)(**constructor)
            method=('odeint' if args.variant=='w' else 'pert2_shanks') if mode=='legacy' else ('picard' if args.variant=='si' else 'picard_table')
            if mode=='reference':
                module.TidalStrippingSolver.subhalo_mass_stripped=independent_solve
                method='dop853'
            else:
                module.TidalStrippingSolver.subhalo_mass_stripped=original
            call=model.rs_rhos_calc if args.variant=='w' else model.subhalo_properties_calc
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always',RuntimeWarning)
                arrays=call(**config,method=method,**(picard_options if mode=='picard' else {}))
            seconds=time.perf_counter()-start
            path=args.output.with_name(args.output.stem+'-'+mode+'.npz')
            np.savez_compressed(path,**{f'tuple_{i}':value for i,value in enumerate(arrays)})
            nonfinite={i:int(np.sum(~np.isfinite(value))) for i,value in enumerate(arrays) if not np.all(np.isfinite(value))}
            if nonfinite:
                raise ValueError(f'Nonfinite catalog fields: {nonfinite}')
            solver=model.stripping_solver
            report['runs'][mode]={'seconds':seconds,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'warnings':[str(w.message) for w in observed],
                 'fallback_events':getattr(solver,'_picard_events',[]),
                 'table_build_seconds':sum(t.build_seconds for t in getattr(solver,'_picard_tables',{}).values()),
                 'table_count':len(getattr(solver,'_picard_tables',{})),
                 'array_file':path.name,'array_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'metrics':metrics(args.variant,module,model,arrays,config)}
            print(json.dumps({'mode':mode,'seconds':seconds,'table_count':report['runs'][mode]['table_count'],
                 'fallbacks':len(report['runs'][mode]['fallback_events'])}),flush=True)
        except Exception as exc:
            report['failures'].append({'mode':mode,'error':repr(exc),'traceback':traceback.format_exc()})
            print(json.dumps({'mode':mode,'failure':repr(exc)}),flush=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    compare_runs(report,args.output.parent)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
