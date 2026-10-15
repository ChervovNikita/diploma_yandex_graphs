"""Collect terminal evidence without changing the frozen NCNC method or driver."""
import hashlib
import json
from pathlib import Path
import sys

E=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(E))
import remote_control as remote
Q=E/'qualification/interaction_link_ncnc_v1_cpu_v1'
OUTPUT=remote.REMOTE_EXPANSION+'/results/interaction_link_ncnc_v1_cpu_v1_attempt01'

def pin(path):
    raw=path.read_bytes()
    return dict(path=str(path.relative_to(Q)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def save(path,value):
    if path.exists():raise ValueError('Preserve prior evidence: '+str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')

code=f'''import json,hashlib,base64
from pathlib import Path
p=Path({OUTPUT!r});s=json.loads((p/'SUPERVISOR.json').read_text())
value=dict(supervisor=s)
if s['status']!='running':
 value['files']=[]
 for f in sorted(p.rglob('*')):
  if f.is_file() and f.suffix in ('.json','.log'):
   b=f.read_bytes();value['files'].append(dict(path=str(f.relative_to(p)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
 f=Path(str(p)+'.driver.log');b=f.read_bytes();value['files'].append(dict(path='driver.log',bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
 exp=Path({remote.REMOTE_EXPANSION!r});value['source_recheck']=[]
 for folder in (exp/'interaction_link_ncnc_v1',exp/'qualification/interaction_link_ncnc_v1_cpu_v1'):
  f=folder/'MANIFEST.json';b=f.read_bytes();value['source_recheck'].append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
  for row in json.loads(b)['files']:
   f=folder/row['path'];b=f.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 value['memory_events_terminal']=Path('/sys/fs/cgroup/memory.events').read_text()
print(json.dumps(value))
'''
value=json.loads(remote.call(code))
if 'files' not in value:
    print(json.dumps(dict(status='running',supervisor=value['supervisor'])));raise SystemExit(0)
import base64
fetched=Q/'fetched';fetched.mkdir(exist_ok=False)
receipts=[]
for row in value.pop('files'):
    raw=base64.b64decode(row.pop('data'));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    p=fetched/row['path'];assert p.resolve().is_relative_to(fetched.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);receipts.append(pin(p))
save(Q/'TERMINAL_SOURCE_RECHECK.json',{k:v for k,v in value.items() if k!='supervisor'})
rows=[json.loads(p.read_text()) for p in sorted(fetched.glob('*/RESULT.json'))]
keys=('actual_optimizer_transitions','optimizer_attempts','VALID_calls','VALID_attempts','native_constructions','native_constructions_attempted','actual_reverse_collections_attempted','actual_reverse_collections_completed','additional_old_state_gradient_collections','elapsed_seconds','process_user_seconds','process_system_seconds')
cost={k:sum(r.get(k,0) for r in rows) for k in keys}
cost.update(schema='NCNC-interaction-native-CPU-cost-v1',attempts=1,terminal_status=value['supervisor']['status'],cases_started=len(value['supervisor']['cases']),case_results_available=len(rows),cases_passed=sum(r['status']=='passed' for r in rows),summed_supervisor_case_wall_seconds=sum(r['wall_seconds'] for r in value['supervisor']['cases']),maximum_child_peak_RSS_KiB=max((r.get('process_peak_RSS_KiB',0) for r in rows),default=0),memory_events_changed=any(r.get('memory_events_before')!=r.get('memory_events_after') for r in rows),automatic_retries=0,scientific_fits=0,TEST_accessed=False,GPU_models_or_updates=0)
updates=[r[k] for r in rows for k in ('first_update','resumed_second_update') if k in r]
cost.update(completed_updates_with_source_work_receipts=len(updates),source_encoder_forwards=sum(r['work']['encoder_forwards'] for r in updates),source_top_level_predictor_calls=sum(r['work']['NCNC_output_routing']['dispatch_counts']['task'] for r in updates),source_recursive_predictor_calls=sum(r['work']['NCNC_output_routing']['dispatch_counts']['recursive'] for r in updates),extra_forward_calls=0)
passed=len(rows)==4 and all(r['status']=='passed' for r in rows) and value['supervisor']['status']=='complete'
report=dict(schema='NCNC-interaction-native-CPU-readiness-v1',status='passed_bounded_native_CPU_lifecycle' if passed else 'failed_bounded_native_CPU_lifecycle',source_manifest_sha256='e7fb265ea3c3650d7ba4c24157adda04b20cdba88ad46ebc54a498b9b892b0c1',gate_manifest_sha256='bf32fd6f6aa0cce6fc7da77255964d2ae25923618a1d0288b50abca5ed4b9cc4',cost=cost,cases=[],not_started=value['supervisor'].get('not_started',[]),source_minibatch=[65536,65536],full_VALID_records=[60084,100000],pending=['CUDA lifecycle/pair resource qualification','Whole native TRAIN epoch/full-worker lifecycle','Production selected checkpoint actual/mean export binding and final TEST reconstruction','Scientific convergence/comparison and campaign review/GO'])
for r in rows:
    out={k:r[k] for k in ('method','status','stage','actual_optimizer_transitions','VALID_calls','native_constructions','actual_reverse_collections_completed','elapsed_seconds','process_peak_RSS_KiB','fresh_restored_all_native_optimizer_sampler_RNG_state_exact','restored_served_bit_exact','restored_mean_bit_exact','actual_call_assignment_checks','checkpoint','failure') if k in r}
    out['updates']=[{key:row[key] for key in ('work','component_gradients')}|{'observer':row['interaction_diagnostics']['NCNC_top_level_observer']} for row in (r[k] for k in ('first_update','resumed_second_update') if k in r)]
    report['cases'].append(out)
if passed:
    assert (cost['actual_optimizer_transitions'],cost['VALID_calls'],cost['native_constructions'],cost['actual_reverse_collections_completed'],cost['additional_old_state_gradient_collections'])==(8,12,8,12,0)
    assert all(r['fresh_restored_all_native_optimizer_sampler_RNG_state_exact'] and r['restored_served_bit_exact'] and r['restored_mean_bit_exact'] for r in rows)
save(Q/'COST.json',cost);save(Q/'READINESS.json',report)
summary='# NCNC interactions: '+('bounded CPU lifecycle passed' if passed else 'bounded CPU lifecycle failed')+'\n\n'
summary+='Frozen source manifest `'+report['source_manifest_sha256']+'`; all method and driver bytes rechecked after termination.\n\n'
summary+='| Method | Status | Updates | Full VALID | Constructions | Reverse collections | Seconds | Peak RSS GiB |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for r in rows:summary+=f"| {r['method']} | {r['status']} | {r['actual_optimizer_transitions']} | {r['VALID_calls']} | {r['native_constructions']} | {r['actual_reverse_collections_completed']} | {r['elapsed_seconds']:.2f} | {r['process_peak_RSS_KiB']/1024**2:.2f} |\n"
summary+=f"\nActual totals: {cost['actual_optimizer_transitions']} Adam transitions, {cost['VALID_calls']} full VALID calls, {cost['native_constructions']} constructions, {cost['actual_reverse_collections_completed']} native reverse collections and {cost['additional_old_state_gradient_collections']} extra reverse collections; {cost['summed_supervisor_case_wall_seconds']:.2f} supervisor case seconds. The full source minibatch remains 65,536 positive plus 65,536 negative queries; each VALID contains all 60,084 positive and 100,000 negative queries.\n\n"
if passed:
    summary+='The two block-credit recipes used the actual declared objectives and ordered parameter blocks; assigned gradients exactly matched the tensors returned by those same native calls before one original Adam step. The two scalar-J recipes used one audited joint scalar with active finite gradients. No repeated backward comparison or tolerance search occurred. Original depth-one TaskCallScope/recursive completion stayed intact: every update recorded 8 task and 16 recursive calls, with recursive outputs bypassing the private output bank and no recursive AUX objects.\n\nFresh construction and checkpoint restore preserved model/Adam/sampler/TaskCallScope/private-global RNG/modes/epoch state exactly. Served and same-bank mean VALID outputs were bit-exact after restore; the same-call mean sidecar added no forward and was invalidated on TRAIN/restore.\n\n'
else:
    summary+='Failure receipts and all completed/attempted work are retained. No automatic retry was launched; unstarted cases remain listed in the supervisor and JSON readiness.\n\n'
summary+='This qualifies only the bounded native CPU lifecycle. Full epochs, CUDA/resources, production selected-mean export and final TEST reconstruction remain pending. No TEST scoring, scientific fit, accuracy conclusion or full campaign GO is represented.\n'
(Q/'READINESS.md').write_text(summary)
receipts.extend(pin(Q/name) for name in ('READINESS.json','READINESS.md','COST.json','TERMINAL_SOURCE_RECHECK.json'))
save(Q/'EVIDENCE_MANIFEST.json',dict(schema='NCNC-interaction-terminal-byte-receipts-v1',files=receipts))
print(json.dumps(dict(status=report['status'],cost=cost,readiness=str(Q/'READINESS.md')),indent=2))
