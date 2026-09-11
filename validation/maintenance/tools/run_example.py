"""Execute the original main notebook cells with a disclosed small grid.

Retains imports, tuple unpacking and observable calls. C recursive-table
regeneration is intentionally separate from the numerical maintenance test.
"""
import argparse,ast,hashlib,json,sys,traceback,warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--variant',required=True);args=p.parse_args()
notebook=Path('Examples.ipynb' if args.variant=='w' else 'sample.ipynb')
source=json.loads(notebook.read_text());config=dict(N_ma=32,N_herm=3,N_hermNa=16,dz=.2,zmax=2.)
report={'notebook':notebook.name,'source_sha256':hashlib.sha256(notebook.read_bytes()).hexdigest(),
        'python':sys.version,'grid_overrides':config,'cells':[],
        'other_overrides':['NumPy RNG seed 20260911','F bootstrap repetitions 2','F diagnostic mass plot 1e6..1e16 Msun'],
        'not_executed':[]}
class SmallGrid(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
        if name in ['subhalo_observables','fdm_subhalo_observables','subhalo_properties_calc','rs_rhos_calc','subhalo_distr','N_sat','N_sat_Vthres']:
            values=dict(config)
            if args.variant=='c':values['logmamin']=-6.
            else:values['logmamin']=6. if args.variant!='w' else 8.
            names=set(values);node.keywords=[kw for kw in node.keywords if kw.arg not in names]
            node.keywords += [ast.keyword(arg=k,value=ast.Constant(value=v)) for k,v in values.items()]
        return node
    def visit_Assign(self,node):
        names=[n.id for n in node.targets if isinstance(n,ast.Name)]
        if args.variant=='f' and 'n_bootstrap' in names:node.value=ast.Constant(2)
        if args.variant=='f' and 'M' in names:node.value=ast.parse('np.logspace(6,16,100)*obs_cdm.Msun',mode='eval').body
        return self.generic_visit(node)
namespace={};np.random.seed(20260911)
for index,cell in enumerate(source['cells']):
    if cell['cell_type']!='code':continue
    raw=''.join(cell['source'])
    if args.variant=='c' and ('import boost_iteration' in raw or 'annihilation_boost_factor(n=4)' in raw):
        report['not_executed'].append({'cell':index,'reason':'Generating/reusing recursive boost tables is a separate data product; n=0 observable is exercised.'});continue
    raw='\n'.join(line for line in raw.splitlines() if not line.lstrip().startswith('%'))
    if not raw.strip():continue
    try:
        tree=ast.fix_missing_locations(SmallGrid().visit(ast.parse(raw)))
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            exec(compile(tree,notebook.name,'exec'),namespace)
        report['cells'].append({'index':index,'status':'passed','warnings':[str(w.message) for w in observed]})
        plt.close('all')
    except Exception as exc:
        report['cells'].append({'index':index,'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        break
report['passed']=all(cell['status']=='passed' for cell in report['cells'])
report['itamae_imported']=any(name=='itamae' or name.startswith('itamae.') for name in sys.modules)
Path('validation/maintenance/example-execution.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['passed'] or report['itamae_imported']:raise SystemExit(1)
