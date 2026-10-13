"""Exact frozen launcher audit; source/data byte authentication only."""
import importlib.abc,json,sys
from pathlib import Path
S=Path(__file__).resolve().parent.parent;P=S.parent;sys.path.insert(0,str(S/'runtime'))
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','pandas','scipy','dgl','torch_geometric','torch_sparse','ogb'):raise AssertionError('Numerical import forbidden:'+fullname)
sys.meta_path.insert(0,Deny());import launcher
plan=json.loads((S/'matrix/PLAN.json').read_text());studies,runs,placement,dependencies=launcher.validate_plan(P,plan)
assert len(studies)==2 and len(runs)==189 and len(placement)==189
v={'schema':'literal-sourceports-exact-frozen-launcher-metadata-audit-v1','status':'passed','studies':2,'run_records':len(runs),'numerical_imports':0,'array_payloads_decoded':0,'source_and_data_opaque_bytes_authenticated':True,'actual_source_launcher':str((S/'runtime/launcher.py').relative_to(P)),'scientific_fits':0,'GO':False,'resource_profile_qualified':False};(S/'FROZEN_LAUNCHER_METADATA_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
