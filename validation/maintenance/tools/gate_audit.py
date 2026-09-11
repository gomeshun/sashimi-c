"""Fail closed on the numerical/default acceptance gates."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);args=p.parse_args()
v=args.variant;b=Path('validation/maintenance');g=json.loads((b/'picard-gates.json').read_text())
domain_file='picard-complete-range-domain.json'
d=json.loads((b/domain_file).read_text());benchmark=json.loads((b/'picard-benchmark.json').read_text())
checks=[]
def check(name,success,details):
 checks.append(dict(name=name,passed=bool(success),details=details))
check('independent mass accuracy',d['maximum_mass_relative_error']<=g['mass_relative_max'],d['maximum_mass_relative_error'])
check('independent reference convergence',d['maximum_reference_refinement_error']<=g['reference']['refinement_relative_max'],d['maximum_reference_refinement_error'])
expected=31 if v=='w' else 45
check('frozen supported domain completed',len(d['cases'])==expected and (len(d['domain_failures'])==1 if v=='w' else not d['domain_failures']),{'cases':len(d['cases']),'domain_failures':d['domain_failures']})
check('whole catalog median speedup',benchmark['whole_catalog_speedup']>=g['performance']['full_catalog_speedup_median_min'],benchmark['whole_catalog_speedup'])
check('three process repetitions',all(len(x)==3 for x in benchmark['runs'].values()),benchmark['repetitions'])
check('ordinary catalog has no fallback',all(not r['catalog_fallback_events'] for r in benchmark['runs']['picard']),[r['catalog_fallback_events'] for r in benchmark['runs']['picard']])
files=list(b.glob('picard-scenario-*-result.json'))+[b/'picard-grid-refinement-result.json',b/('picard-accepted-catalog.json' if v=='w' else 'picard-final-catalog.json')]
maximum_metric=0.;maximum_mask_weight=0.;maximum_catalog_mass=0.
for path in files:
 data=json.loads(path.read_text());check(path.name+' completed',not data['failures'],data['failures'])
 c=data['comparison'];maximum_catalog_mass=max(maximum_catalog_mass,c['maximum_mass_relative_error'])
 for name,metric in c['metric_differences'].items():
  if metric['relative'] is not None:maximum_metric=max(maximum_metric,metric['relative'])
  else:check(path.name+' '+name+' zero reference',metric['absolute']==0,metric['absolute'])
 maximum_mask_weight=max(maximum_mask_weight,max(x['weight_fraction'] for x in c['changed_masks'].values()))
 check(path.name+' mass function',c['mass_function_l1_fraction']<=g['smooth_catalog_relative_max'],c['mass_function_l1_fraction'])
check('catalog mass error',maximum_catalog_mass<=g['mass_relative_max'],maximum_catalog_mass)
check('smooth catalog observables',maximum_metric<=g['smooth_catalog_relative_max'],maximum_metric)
check('survival masks',maximum_mask_weight<=g['boundary']['changed_survival_or_collapse_weight_fraction_max'],maximum_mask_weight)
boundary=json.loads((b/'picard-boundaries.json').read_text())
check('explicit survival and collapse boundary audit',all(item['changed_weight_fraction']<=g['boundary']['changed_survival_or_collapse_weight_fraction_max'] for case in boundary['cases'] for item in case['boundaries']),{'catalogs':len(boundary['cases']),'changed_nodes':sum(len(item['changed_nodes']) for case in boundary['cases'] for item in case['boundaries'])})
check('standalone numerical helper present',Path('picard_tidal_stripping.py').is_file(),hashlib.sha256(Path('picard_tidal_stripping.py').read_bytes()).hexdigest())
report={'variant':v,'gate_specification':g['specification'],'passed':all(c['passed'] for c in checks),'checks':checks,
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py') if not p.name.startswith('test')},
 'limits':'Numerical agreement and runtime evidence only. No simulation calibration, downstream inference, main merge, or publication.'}
(b/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'variant':v,'passed':report['passed'],'failed':[c for c in checks if not c['passed']]}))
if not report['passed']:raise SystemExit(1)
