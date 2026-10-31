"""Stage four disabled Bayesian resource cases; metadata only, no callback."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,invoke

REMOTE=GUARD+r'''
import gzip
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
out=phase/payload['destination'];assert not out.exists();out.mkdir()
def receipt(path):return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for row in payload['files']:
 path=out/row['name'];raw=base64.b64decode(row['data'])
 assert path.parent==out and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path.write_bytes(raw)
packet=phase/'native_bsnn_bayesian_newdata_packet_20261012_v3'
plan=json.loads((packet/'matrix/PLAN.json').read_text())
custody_path=phase/'native_nsd_five_new_data_packet_20261012_v2/CUSTODY.json'
custody=json.loads(custody_path.read_text())
assert custody['dataset']=='twitch-de-musae-raw' and custody['model_graph']=={'nodes':9498,'features':3170,'classes':2,'edges':306276}
cases={};forbidden=set();source={}
source[str(custody_path.relative_to(phase))]=receipt(custody_path)
for role in custody['splits']:
 forbidden.add(role['node_inputs']['select']['path'])
 for label in ('cal','evaluate'):forbidden.add(role['endpoint_inputs'][label+'_targets']['path'])
def collect(value):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=set(value):
   p=phase/value['path'];assert p.resolve().is_relative_to(phase) and not p.is_symlink()
   actual=receipt(p);assert actual=={k:value[k] for k in ('path','bytes','sha256')}
   source[actual['path']]=actual
  else:
   for v in value.values():collect(v)
 elif isinstance(value,list):
  for v in value:collect(v)
def targets(value,key=''):
 if isinstance(value,dict):
  if key in ('select','cal','evaluate','cal_targets','evaluate_targets','select_targets') and 'path' in value:forbidden.add(value['path'])
  for k,v in value.items():targets(v,k)
 elif isinstance(value,list):
  for v in value:targets(v,key)
for item in plan['studies']:
 study=json.loads(gzip.decompress((phase/item['path']).read_bytes()));targets(study)
 assert study['dataset']=='twitch-de-musae-raw'
 for arm in ('native_M1','original_GNNM'):
  run=next(r for r in study['runs'] if r['kind']=='model_fit' and r['config']['bayesian_bsnn_arm']==arm)
  c=run['config'];variant=c['bayesian_bsnn_variant'];B=c['bayesian_bsnn_sources'];collect(B)
  protocol=json.loads((phase/B['protocols'][variant]['path']).read_text())
  args={**protocol['native_args'],'input_dim':3170,'output_dim':2,'graph_size':9498,'device':'cuda:0'}
  role=next(v for v in custody['splits'] if v['node_inputs']['train']==c['node_inputs']['train'])
  n=role['counts']['train'];assert type(n) is int and n>0
  for label in ('graph','train'):collect(c['node_inputs'][label])
  cases[variant+'__'+arm]={'runtime':str((packet/'runtime').relative_to(phase)),
   'run':run,'native_args':args,'TRAIN_rows':n,'full_nodes':9498,'raw_features':3170,'directed_edges':306276,
   'prospective_fit_source_scope':study['study_id'],'fixed_native_tuple_no_HPO':True,'all_10_role_graphs_equal':True,
   'TRAIN_rows_admission_note':'Root binds count from declared source custody; not from target decoding.'}
assert len(cases)==4
for p in sorted((packet/'runtime').rglob('*.py')):source[str(p.relative_to(phase))]=receipt(p)
template=json.loads((out/'ROOT_RELEASE_TEMPLATE.json').read_text());assert template['execution_enabled'] is False
roster={'schema':'four-Bayesian-BSNN-full-input-resource-cases-v1','cases':cases,'scientific_fits':0,'optimizer_updates':0}
(out/'CASES.json').write_text(json.dumps(roster,indent=2,sort_keys=True)+'\n')
template.update(cases_pin={k:v for k,v in receipt(out/'CASES.json').items() if k!='path'},
 source_files=list(source.values()),forbidden_target_paths=sorted(forbidden))
(out/'ROOT_RELEASE_TEMPLATE_WITH_SOURCE_PINS.json').write_text(json.dumps(template,indent=2,sort_keys=True)+'\n')
manifest={'schema':'four-Bayesian-BSNN-root-resource-proposal-source-manifest-v1','files':[receipt(p) for p in sorted(out.iterdir()) if p.is_file()],
 'actual_input_callbacks_executed':0,'scientific_fits':0,'optimizer_updates':0,'execution_enabled':False,'root_read_and_admission_pending':True,'GO':False}
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for p in out.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'manifest':receipt(out/'MANIFEST.json'),'cases':roster,'release_template':template,
 'scientific_fits':0,'optimizer_updates':0,'source_callbacks':0,'GO':False}))
'''

def main():
    files=[]
    for name in ('qualify_bsnn.py','stage_admission_metadata.py','ROOT_RELEASE_TEMPLATE.json','ADMISSION_PROPOSAL.json','READ_BEFORE_ADMISSION.md','SOURCE_PARSE_QA.json'):
        path=HERE/name
        raw=path.read_bytes()
        if path.suffix=='.py':ast.parse(raw)
        files.append({'name':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'destination':HERE.name,'files':files}).encode(),9)).decode()
    assert len(encoded)<48000
    result=invoke(REMOTE,[encoded],HERE,'STAGE_SOURCE',180)
    assert result['exit_code']==0,'Preserve staging failure before a separate successor'
    value=json.loads(result['stdout'])
    for name,key in [('CASES.json','cases'),('ROOT_RELEASE_TEMPLATE_WITH_SOURCE_PINS.json','release_template')]:
        (HERE/name).write_text(json.dumps(value[key],indent=2,sort_keys=True)+'\n')
    (HERE/'ALLOCATION_HANDOFF.json').write_text(json.dumps({k:v for k,v in value.items() if k not in ('cases','release_template')},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest':value['manifest'],'cases':[(k,v['TRAIN_rows']) for k,v in value['cases']['cases'].items()],
                     'source_files':len(value['release_template']['source_files']),'scientific_fits':0}))

if __name__=='__main__':main()
