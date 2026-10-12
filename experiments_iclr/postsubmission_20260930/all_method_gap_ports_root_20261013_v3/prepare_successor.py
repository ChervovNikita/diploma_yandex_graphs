"""Repair source-owned Single8 state forwarding in an immutable successor."""
import base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent
sys.path.insert(0,str(P/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke
REMOTE=GUARD+r'''
import copy,gzip,importlib
v=json.loads(base64.b64decode(sys.argv[1]));out=phase/'all_method_gap_ports_root_20261013_v3';assert not out.exists();out.mkdir();(out/'runtime').mkdir();helper=out/'runtime/plain_numpy_state.py';raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];helper.write_bytes(raw);compile(raw,str(helper),'exec')
sys.path.insert(0,str(phase/'master_controller_source_20261013_v23/runtime'));import spec,compile_master_plan as master
entries=json.loads((phase/'all_method_gap_ports_root_20261013_v2/MASTER_INPUT_ENTRIES.json').read_text());result=[]
def receipt(p):
 b=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
for e in entries:
 if e['id']=='literal_complete_private_hop_source2000':result.append(e);continue
 plan=json.loads((phase/e['plan']['path']).read_text());matrix=out/Path(e['plan']['path']).parent.name;matrix.mkdir();(matrix/'studies').mkdir();studies=[];changed=0
 for ref in plan['studies']:
  s=json.loads(gzip.decompress((phase/ref['path']).read_bytes()))
  for r in s['runs']:
   if r['kind']=='model_fit' and 'plain_numpy_helper' in r['config']:r['config']['plain_numpy_helper']=str(helper.relative_to(phase));changed+=1
  files={r['path']:r for r in s['source']['files']};files[str(helper.relative_to(phase))]=receipt(helper);s['source']['files']=[files[k] for k in sorted(files)];spec.validate(s)
  q=matrix/'studies'/(s['study_id']+'.json.gz');q.write_bytes(gzip.compress((spec.canonical(s)+'\n').encode(),mtime=0));studies.append((s,q))
 plan.update(studies=[{'path':str(q.relative_to(phase)),'spec_hash':spec.fingerprint(s)} for s,q in studies],output_root=str(matrix.relative_to(phase))+'/execution')
 for gate in plan['global_gates'].values():gate['path']=str(matrix.relative_to(phase))+'/gate_receipts/'+Path(gate['path']).name
 summary=json.loads((phase/e['summary']['path']).read_text());summary.update(plan_hash=spec.fingerprint(plan),exact_actual_controller_counts=master.typed_counts([s for s,q in studies]),Single8_source_clock_forwarding_repaired=True,source_learning_unchanged=True,scientific_fits=0,GO=False)
 for n,value in [('PLAN.json',plan),('SUMMARY.json',summary)]:(matrix/n).write_text(spec.canonical(value)+'\n')
 lineage={'prior_entry':e,'changed_fit_helper_bindings':changed,'source_operator_unchanged':True,'source_only_delegated_restore_repair':True,'fits':0,'GO':False};(matrix/'LINEAGE.json').write_text(spec.canonical(lineage)+'\n')
 entry=dict(e,plan=receipt(matrix/'PLAN.json'),summary=receipt(matrix/'SUMMARY.json'),plan_hash=spec.fingerprint(plan),coverage=e['coverage']+[receipt(matrix/'LINEAGE.json')]);(matrix/'MASTER_INPUT_ENTRY.json').write_text(spec.canonical(entry)+'\n');result.append(entry)
(out/'MASTER_INPUT_ENTRIES.json').write_text(spec.canonical(result)+'\n');print(json.dumps({'entries':result,'source_helper':receipt(helper),'scientific_fits':0,'GO':False}))
'''
def main():
 out=HERE/'prepare_01';out.mkdir();raw=(HERE/'runtime/plain_numpy_state.py').read_bytes();v={'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()};r=invoke(REMOTE,[base64.b64encode(json.dumps(v).encode()).decode()],out,'PREPARE',120);assert r['exit_code']==0
 x=json.loads(r['stdout']);(out/'RESULT.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'entries':len(x['entries']),'fits':0,'GO':False}))
if __name__=='__main__':main()
