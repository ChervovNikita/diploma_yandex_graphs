"""Inspect named disabled metadata on the verified allocation; no role arrays."""
import gc
import json
from pathlib import Path

from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent / 'root_master_inspection_20261012_v1'
REMOTE = GUARD + r'''
import gc,gzip,importlib
runtime=phase/'training_pipeline_v2_20261012/qa/cpu_detached_startup_v21'
sys.path.insert(0,str(runtime))
master=importlib.import_module('compile_master_plan')
out=phase/'master_input_inspection_20261012_v1';assert not out.exists();out.mkdir()
def save(name,value):
 (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
rows=[]
entries=[]
for identifier,relative in (
 ('native_v3','massive_native_packet_20261012_v3/matrix'),
 ('coordinate_v1','massive_coordinate_packet_20261012_v1/matrix'),
 ('molecular_v4','massive_molecular_packet_20261012_v4/matrix'),
 ('polyformer_v1','massive_polyformer_packet_20261012_v1/matrix'),
 ('polyformer_author_v1','massive_polyformer_author_schedule_packet_20261012_v1/matrix'),
 ('link_v1','link_plan_packet_20261012_v1/matrix'),
 ('context_keys_v1','context_keys_packet_20261012_v1')):
 entry={'id':identifier,'plan':master.receipt(phase,relative+'/PLAN.json'),
        'summary':master.receipt(phase,relative+'/SUMMARY.json')}
 entries.append(entry)
 row={'id':identifier,'input':entry,'status':'pending_inspection'}
 try:
  packet=master.load_packet(phase,entry)
  row.update(status='loaded_disabled_metadata',study_count=len(packet['studies']),counts=packet['counts'])
  examples={}
  for study in packet['studies']:
   for run in study['runs']:
    if run['kind']!='model_fit':continue
    config=run['config']
    label=str((config.get('architecture'),config.get('source_arm'),config.get('mode'),config.get('arm')))
    if label not in examples:
     examples[label]={'study_id':study['study_id'],'run_id':run['run_id'],'budget':run['budget'],
        'config':config,'seed':run['seed'],'body':run['body']}
  row['ownership_examples']=examples
  del packet
 except Exception as error:
  row.update(status='blocked_metadata',error_type=type(error).__name__,error=str(error))
  plan=json.loads((phase/relative/'PLAN.json').read_text())
  if plan['studies']:
   first=plan['studies'][0];raw=(phase/first['path']).read_bytes()
   study=json.loads(gzip.decompress(raw) if first['path'].endswith('.gz') else raw)
   runmap={run['run_id']:run for run in study['runs']}
   row['first_fit_examples']=[]
   labels=set()
   for run in study['runs']:
    config=run['config']
    if run['kind']=='model_fit':
     label=str((config.get('architecture'),config.get('source_arm'),config.get('mode'),config.get('arm')))
     if label not in labels:
      labels.add(label);row['first_fit_examples'].append({'run_id':run['run_id'],
         'config':config,'budget':run['budget'],'seed':run['seed'],'body':run['body']})
   for run in study['runs']:
    parents=run['config'].get('fit_parents',[])
    if run['kind']=='ensemble_assembly' and len(parents)==4:
     configs=[runmap[p]['config'] for p in parents]
     row['first_I4_diff']={key:[config.get(key) for config in configs]
        for key in set().union(*(config.keys() for config in configs))
        if any(config.get(key)!=configs[0].get(key) for config in configs[1:])}
     break
   del study,runmap
 rows.append(row);save('INSPECTION.json',{'rows':rows,'scientific_fits':0,'role_arrays_opened':0})
 gc.collect()
save('INPUT_INDEX.json',{'schema':'disabled-master-input-index-v1','packets':entries})
publication=phase/'publication/Closed_resource_native_coordinate_author_controller_20261012_v1'
push={name:json.loads(path.read_text()) for name in ('PUSH_RECEIPT_SSH22_V1.json','PUSH_RECEIPT.json')
      if (path:=publication/name).exists()}
save('PUBLICATION_RECEIPTS.json',push)
print(json.dumps({'inspection':rows,'publication_receipts':push,'scientific_fits':0,'role_arrays_opened':0}))
'''


def main():
    HERE.mkdir(exist_ok=True)
    assert not (HERE / 'inspect_named_packets.json').exists()
    transport = invoke(REMOTE, [], HERE, 'inspect_named_packets', timeout=600)
    if transport['exit_code'] != 0:
        raise RuntimeError('Metadata inspection failed; preserved transport receipt')
    value = json.loads(transport['stdout'])
    (HERE / 'INSPECTION.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'packets': [{k: row.get(k) for k in ('id', 'status', 'error')}
                                 for row in value['inspection']],
                      'scientific_fits': 0, 'role_arrays_opened': 0}))


if __name__ == '__main__':
    main()
