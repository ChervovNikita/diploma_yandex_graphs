"""V2 successor from bound source/descriptor metadata only; never numerical work."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
V1 = P / 'common_wrapper_graph_reliability_source_20261010_v1'
ROOT = P / 'common_wrapper_graph_reliability_root_20261010_v1'
REMOTE_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
if (D / 'SEAL.json').exists():
    raise SystemExit('Preserve sealed source; use a successor.')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value): (D/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
support = json.loads((V1/'FROZEN_SUPPORT.json').read_text())
receipt = json.loads((ROOT/'FAILED_PREFIX_CACHE_BINDINGS.json').read_text())
assert len(receipt['records']) == 4 and receipt['new_forwards'] == 0 and receipt['labels_opened'] is False
for row in receipt['records']:
    local = ROOT / (row['record']['key']+'.json')
    assert digest(local) == row['sha256'] and local.stat().st_size == row['bytes']
    assert json.loads(local.read_text()) == row['record']
    assert row['record']['discrepancies']['max_abs_logit_difference'] <= 1e-4
assert (D/'operators.py').read_bytes() == (V1/'operators.py').read_bytes()
assert (D/'study.py').read_bytes() == (V1/'study.py').read_bytes()
support['UTC'] = UTC
support['source_files'] = [{'path':str(q.relative_to(P)),'sha256':digest(q)}
                           for q in sorted(D.glob('*.py')) if q.name not in ('BUILD_SUPPORT.py','check_static.py')]
support['comparison_tolerance'] = {'max_absolute_logit_error':1e-4,'relative_error_acceptance':False,
                                   'argmax_differences_recorded_not_used_for_acceptance':True,
                                   'own_node_logits_remain_archived_authority':True,
                                   'descriptor_only_acceptance':'max_abs_logit_difference<=1e-4',
                                   'investigation_or_replay_loop':False}
support['prefix_reuse'] = {'records':receipt['records'],'prior_owner_end':receipt['prior_owner_end'],
                          'owner_end_path':REMOTE_PHASE+'/common_wrapper_graph_reliability_root_20261010_v1/OWNER_END.json',
                          'old_manifest_path':REMOTE_PHASE+'/common_wrapper_graph_reliability_source_20261010_v1/MANIFEST.json',
                          'old_manifest_sha256':digest(V1/'MANIFEST.json'),
                          'old_support_path':REMOTE_PHASE+'/common_wrapper_graph_reliability_source_20261010_v1/FROZEN_SUPPORT.json',
                          'old_support_sha256':digest(V1/'FROZEN_SUPPORT.json'),
                          'prior_forward_calls':10,'prior_member_trajectories':10,'new_forward_calls':89,
                          'new_member_trajectories':116,'represented_forward_calls':99,'represented_member_trajectories':126,
                          'reused_banks':4,'new_export_banks':41,'old_artifacts_edited':False}
support['unchanged_root_development_criteria'] = {'path':REMOTE_PHASE+'/common_wrapper_graph_reliability_root_20261010_v1/DECISION.md',
                                               'sha256':digest(ROOT/'DECISION.md'),'amended':False}
write('FROZEN_SUPPORT.json',support)
write('PREFIX_REUSE.json',support['prefix_reuse'])
old_bindings = json.loads((V1/'SOURCE_BINDINGS.json').read_text())['inputs']
new_paths = [V1/'MANIFEST.json',V1/'SEAL.json',V1/'FROZEN_SUPPORT.json',V1/'operators.py',V1/'study.py',
             ROOT/'FAILED_PREFIX_CACHE_BINDINGS.json',ROOT/'FAILED_BANK_RECORD.json',ROOT/'DECISION.md']
new_paths += [ROOT/(r['record']['key']+'.json') for r in receipt['records']]
write('SOURCE_BINDINGS.json', {'UTC':UTC,'inputs':old_bindings+[{'path':str(q.relative_to(P)),'bytes':q.stat().st_size,
                            'sha256':digest(q),'scope':'V1 immutable source or descriptor/closure metadata only.'} for q in new_paths],
                            'model_dataset_label_logit_prediction_or_cache_payload_reads':False,
                            'numerical_calls_remote_commands_or_fits':False})
print(json.dumps({'v2_support':str(D/'FROZEN_SUPPORT.json'),'four_prefix_descriptors_bound':True,
                  'new_forward_calls_if_root_admits':89,'new_member_trajectories':116,
                  'represented_calls':99,'represented_trajectories':126,'numerical_execution':False}))
