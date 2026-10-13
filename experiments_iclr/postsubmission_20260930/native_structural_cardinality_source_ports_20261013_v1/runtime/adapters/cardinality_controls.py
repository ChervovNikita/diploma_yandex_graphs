"""Fixed C64 anchor controls use the unchanged native boundary runner."""
import copy
from . import link_boundary

def make_adapter(*,torch,project_root,study,run,device):
 from structural_cardinality_semantics import validate_cardinality_control
 validate_cardinality_control(run);clone=copy.deepcopy(study);clone['runs']=[r for r in clone['runs'] if r['kind']!='model_fit' or 'cardinality_control_contract' in r['config']]
 for row in clone['runs']:
  for name in ('cardinality_control','cardinality_control_contract'):row['config'].pop(name,None)
 actual=copy.deepcopy(run)
 for name in ('cardinality_control','cardinality_control_contract'):actual['config'].pop(name,None)
 return link_boundary.make_adapter(torch=torch,project_root=project_root,study=clone,run=actual,device=device)
