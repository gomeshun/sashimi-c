"""Separate prescribed corrections from the subsequent solver change."""
from pathlib import Path
import argparse,importlib,json,hashlib,warnings
import numpy as np
from catalog_pilot import metrics
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);args=p.parse_args()
v=args.variant;module=importlib.import_module('sashimi_'+v);base=Path('validation/maintenance')
config=json.loads((base/'A-config.json').read_text());params=config['parameters'].copy()
cls=getattr(module,config['class']);saved=np.load(base/'A.npz');n=27 if v=='si' else 10
old=[saved[f'tuple_{i}'] for i in range(n)];out={'A':old};report={'variant':v,'A_source_revision':config['source_revision'],
 'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'parameters':params,'stages':{},'changes':{}}
for name,method in [('corrected_legacy','odeint' if v=='w' else 'pert2_shanks'),('corrected_picard','picard' if v=='si' else 'picard_table')]:
 model=cls(**config.get('constructor',{}));params['method']=method
 with warnings.catch_warnings(record=True) as observed:
  arrays=getattr(model,config['method'])(**params)
 assert all(np.all(np.isfinite(x)) for x in arrays)
 target=base/('three-stage-'+name+'.npz');np.savez_compressed(target,**{f'tuple_{i}':x for i,x in enumerate(arrays)})
 out[name]=arrays
 report['stages'][name]={'array_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'metrics':metrics(v,module,model,arrays,params),'warnings':[str(w.message) for w in observed]}
for first,last in [('A','corrected_legacy'),('corrected_legacy','corrected_picard')]:
 fields=[]
 for i,(a,b) in enumerate(zip(out[first],out[last])):
  a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
  finite=np.isfinite(a)&np.isfinite(b);nonzero=finite&(a!=0)
  fields.append({'field':i,'maximum_absolute_difference':float(np.max(np.abs(a[finite]-b[finite]))) if np.any(finite) else None,
   'maximum_relative_difference':float(np.max(np.abs(b[nonzero]/a[nonzero]-1))) if np.any(nonzero) else None,
   'changed_entries':int(np.sum(a!=b)),'nonfinite_before':int(np.sum(~np.isfinite(a))),'nonfinite_after':int(np.sum(~np.isfinite(b)))})
 report['changes'][first+' -> '+last]=fields
(base/'three-stage-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
