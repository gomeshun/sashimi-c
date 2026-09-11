"""Audit physical survival/collapse boundaries from saved catalogs."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--variant',required=True);args=p.parse_args()
base=Path('validation/maintenance')
report={'variant':args.variant,'cases':[]}
for source in sorted(list(base.glob('picard-scenario-*-result.json')) + list(base.glob('picard-grid-refinement-result.json')) + list(base.glob('picard-final-catalog.json')) + list(base.glob('picard-accepted-catalog.json'))):
 d=json.loads(source.read_text())
 a=np.load(base/d['runs']['picard']['array_file']);b=np.load(base/d['runs']['reference']['array_file'])
 wi,ci=(24,21) if args.variant=='si' else (8,7)
 weights=np.maximum(a[f'tuple_{wi}'],b[f'tuple_{wi}'])
 diagnostics=[('survival',ci,d['config'].get('ct_th',0.))]
 if args.variant=='si': diagnostics.append(('collapse',22,1.1))
 # SIDM tt_th is the unchanged native model clipping threshold.
 if args.variant=='si':
  import sashimi_si
  diagnostics[-1]=('collapse',22,sashimi_si.SIDM_parametric_model().tt_th)
 row={'catalog':source.name,'boundaries':[]}
 for name,index,threshold in diagnostics:
  x,y=a[f'tuple_{index}'],b[f'tuple_{index}']
  changed=(x>threshold)!=(y>threshold) if name=='survival' else (x>=threshold)!=(y>=threshold)
  valid=weights>0
  nearest=np.argsort(np.where(valid,abs(y-threshold),np.inf))[:5]
  row['boundaries'].append({'name':name,'threshold':threshold,
      'changed_weight_fraction':float(np.sum(weights[changed])/np.sum(weights)) if np.sum(weights) else 0.,
      'changed_nodes':[{'index':int(j),'reference':float(y[j]),'picard':float(x[j]),'weight':float(weights[j])} for j in np.flatnonzero(changed)],
      'nearest_positive_weight_nodes':[{'index':int(j),'ma_msun':float(b['tuple_0'][j]),'z_acc':float(b['tuple_1'][j]),'reference':float(y[j]),'picard':float(x[j]),'weight':float(weights[j])} for j in nearest if valid[j]]})
 report['cases'].append(row)
(base/'picard-boundaries.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
