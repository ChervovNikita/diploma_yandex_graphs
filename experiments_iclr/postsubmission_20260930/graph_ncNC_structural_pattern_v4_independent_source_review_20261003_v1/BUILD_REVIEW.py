"""Independent metadata custody verification only; no project imports/execution."""
from pathlib import Path
import ast, datetime, hashlib, json

P=Path(__file__).resolve().parent
R=P.parent
C=R/'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4'
V3=R/'graph_ncNC_structural_pattern_pilot_preparation_20261003_v3'
N=R/'graph_ncNC_member_completion_qualification_preparation_20261003_v2'
UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)}
def save(n,x): (P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def verify(d,pin):
    assert sha(d/'MANIFEST.json')==pin
    m=json.loads((d/'MANIFEST.json').read_text())
    for row in m['files']:
        f=(d/row['path']).resolve()
        assert f.is_relative_to(d.resolve())
        assert f.stat().st_size==row.get('bytes',row.get('size')) and sha(f)==row['sha256']
    return m

cm=verify(C,'9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c')
assert sha(C/'SEAL.json')=='beffda0bbdbb3d71a232671349bb40b4ff1dec03b40c6ef81706b5ee287a19e3'
vm=verify(V3,'fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89')
inputs={'candidate_manifest':ref(C/'MANIFEST.json'),'candidate_seal':ref(C/'SEAL.json'),
        'predecessor_manifest':ref(V3/'MANIFEST.json'),'predecessor_seal':ref(V3/'SEAL.json'),
        'native_prototype':ref(N/'prototype.py'),'native_graph_ops':ref(N/'graph_ops.py'),
        'runtime_authority':ref(R/'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json'),
        'upstream_checkpoint_source':ref(P/'upstream/torch_checkpoint.py')}
save('INPUT_BINDINGS.json',{'schema':'independent_V4_source_review_inputs_v1','UTC':UTC,'inputs':inputs,
     'upstream_commit':'e2d141dbde55c2a4370fac5165b0561b6af4798b',
     'upstream_is_not_installed_remote_file_inspection':True,'candidate_or_predecessor_modified':False})

complete_names=['pattern_model.py','checkpoint_parity.py','pattern_qualification.py','pilot_common.py',
 'pattern_train.py','pilot_state.py','pattern_checks.py','pattern_run.py','pilot_accounting.py',
 'V3_TO_V4.diff','DEPENDENCIES.json','ACTIVATION_CHECKPOINT_POLICY.json','V4_ENGINEERING_QUALIFICATION_PLAN.json',
 'V4_SOURCE_DIFF_EVIDENCE.json','PRESERVED_FULL_GRAPH_FAILURES.json','README.md']
scopes=[{'source':ref(C/n),'scope':'complete substantive file text read'} for n in complete_names]
scopes += [{'source':ref(N/n),'scope':'complete substantive source file text read'} for n in ['prototype.py','graph_ops.py']]
scopes += [{'source':ref(C/'pattern_diagnostics.py'),'scope':'complete score_valid_routes function and preceding load_fit; partial selected_valid exposure','line_ranges':[[1,57]]},
 {'source':ref(C/'pilot_model.py'),'scope':'selected rg context for module binding, one-device runtime admission and seeding; other bodies hash/AST checked','line_ranges':[[9,15],[30,43],[73,78],[100,108]]},
 {'source':ref(C/'pattern_fit.py'),'scope':'keyword navigation only; exact byte-identity to immutable V3 checked'},
 {'source':ref(C/'pilot_evaluate.py'),'scope':'keyword navigation only; exact byte-identity to immutable V3 checked'},
 {'source':ref(P/'upstream/torch_checkpoint.py'),'scope':'selected official source, no import/execution','line_ranges':[[115,204],[343,504],[735,820],[1032,1158],[1430,len((P/'upstream/torch_checkpoint.py').read_text().splitlines())]]},
 {'source':ref(C/'PREPARATION_RESULT.json'),'scope':'complete preparation-result metadata; not used as proof'},
 {'source':ref(C/'ROOT_RELEASE_EXAMPLE.json'),'scope':'complete disabled example metadata; not an authorization'},
 {'source':ref(C/'SEAL.json'),'scope':'complete seal metadata; all payload bytes independently verified'},
 {'source':ref(R/'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json'),
  'scope':'selected version/device/profile/source-pin fields displayed; whole authority JSON hash checked; output included truncation'}]
save('READ_SCOPES.json',{'schema':'independent_V4_review_read_scopes_v1','UTC':UTC,'scopes':scopes,
     'manifest_byte_verification_is_not_semantic_read':True,'source_AST_compile_is_not_runtime':True,
     'prepared_static_checks_not_used_as_oracle':True,'full_upstream_source_audit':False,
     'remote_installed_Torch_checkpoint_source_read':False})

fail=json.loads((C/'PRESERVED_FULL_GRAPH_FAILURES.json').read_text())
rows=[]
for f in fail['failures']:
    for e in f['evidence']:
        for label in ['original','preserved_copy']:
            x=e[label];p=R/x['path'];assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
    term=json.loads((R/f['evidence'][0]['preserved_copy']['path']).read_text())
    peaks=term['CUDA_peak_observation']
    assert term['child_exit_code']==f['child_exit_code']==88
    assert term['supervisor_inclusive_wall_seconds']==f['paid_supervisor_inclusive_wall_seconds']
    assert peaks['cuda_peak_allocated_bytes']==f['CUDA_allocated_peak_bytes']
    assert peaks['cuda_peak_reserved_bytes']==f['CUDA_reserved_peak_bytes']
    assert peaks['caps']==f['caps']
    assert term['qualification_receipt'] is None
    attempts=json.loads((R/f['evidence'][2]['preserved_copy']['path']).read_text())
    work=attempts['attempts'][-1]['work']
    assert work['arm']=='J' and work['completed_batches']==0 and work['attempted_batch']==1
    assert work['phase']=='masked_graph_native_and_pattern_forward'
    rows.append({'attempt_version':f['attempt_version'],'physical_terminal':f['evidence'][0]['preserved_copy'],
                 'supervisor_inclusive_wall_seconds':term['supervisor_inclusive_wall_seconds'],
                 'peak_allocated_bytes':peaks['cuda_peak_allocated_bytes'],'peak_reserved_bytes':peaks['cuda_peak_reserved_bytes'],
                 'caps':peaks['caps'],'child_exit_code':88,'qualification_receipt':None,'work_at_failure':work})
assert sum(x['supervisor_inclusive_wall_seconds'] for x in rows)==fail['paid_wall_sum_seconds']
assert rows[1]['caps']==json.loads((C/'ACTIVATION_CHECKPOINT_POLICY.json').read_text())['physical_caps_ceiling_retained_from_prior_candidate']
save('FAILURE_CUSTODY_CHECK.json',{'schema':'independent_original_failure_value_check_v1','UTC':UTC,
    'original_and_copy_hash_size_checks':20,'value_checks_against_physical_terminal_authority':True,
    'attempts':rows,'paid_wall_sum_seconds':fail['paid_wall_sum_seconds'],'candidate_ceilings_equal_second_attempt':True,
    'zero_first_J_updates_supported_by_attempt_progress':True,'model_quality_or_graph_metric_from_failures':False})

findings=[
 {'id':'checkpoint_scope','status':'NO_MATERIAL_SOURCE_BLOCKER','evidence':['pattern_model.py:11-22','prototype.py:168-173'],
  'argument':'Nonreentrant complete native depth_zero replay with explicit member/tensor/graph/query inputs; no new chunk or empty shortcut.'},
 {'id':'mutable_flags_and_autograd','status':'NO_MATERIAL_SOURCE_BLOCKER','evidence':['pattern_model.py:43-81','prototype.py:115-125','prototype.py:162-173','graph_ops.py'],
  'argument':'Native body does not read cleared capture flags; LayerNorm has no mutable running state; dropout mutates only fresh intermediates; graph operations read inputs; optimizer/flags remain stable until backward completes.'},
 {'id':'RNG_semantics','status':'SOURCE_ARGUMENT_WITH_CURRENT_RUNTIME_GATE','evidence':['upstream/torch_checkpoint.py:131-193','upstream/torch_checkpoint.py:738-766','upstream/torch_checkpoint.py:1490-1525','checkpoint_parity.py'],
  'argument':'Upstream captures forward early-stop and Torch CPU/argument-device RNG, forks/restores caller RNG during recomputation. Single initialized CUDA device and Torch-only randomness match source. Default determinism is metadata only.'},
 {'id':'parity_adequacy','status':'ADEQUATE_PROSPECTIVE_SCOPED_PLAN_UNEXECUTED','evidence':['checkpoint_parity.py','pattern_qualification.py','pilot_common.py:124-145'],
  'argument':'Exact saved V3 pin, sequential 2-update J/F trajectories, raw/target/objective/support/all-gradient/Adam/flags/all-RNG comparisons, main-detach, disabled and empty probes, no state/RNG donation.'},
 {'id':'native_target_and_serving','status':'PRESERVED_AT_SOURCE','evidence':['pattern_model.py','prototype.py:175-220','V3_TO_V4.diff','CUSTODY_AND_DIFF_CHECK.json'],
  'argument':'Inherited direct uncaptured/disabled score path, detached target score, original clamp/private decode/mean logits, unchanged computational helpers and plan.'},
 {'id':'full_work','status':'PRESERVED_WITH_ADDED_PAID_PARITY','evidence':['pattern_qualification.py','pattern_train.py','pattern_diagnostics.py:32-54','pattern_run.py','pilot_accounting.py'],
  'argument':'8 new audits +6 replay +34 native updates=48 full-graph updates, original17-batch epochs and two complete five-route VALID traversals; 100epochs/1700updates/100selectors per fit unchanged; cumulative peaks include probes.'},
 {'id':'paid_failures','status':'PRESERVED_AND_INDEPENDENTLY_VERIFIED','evidence':['FAILURE_CUSTODY_CHECK.json'],
  'argument':'Both exit88 first-J-forward cap failures, exact original/copied bytes and costs, retained second cap ceiling, no predictive result.'},
]
save('REVIEW.json',{'schema':'independent_V4_checkpoint_source_review_v1','UTC':UTC,
    'candidate_manifest_sha256':sha(C/'MANIFEST.json'),'candidate_seal_sha256':sha(C/'SEAL.json'),
    'verdict':'NO_IDENTIFIED_MATERIAL_SOURCE_BLOCKER_IN_SCOPED_CHANGE',
    'source_blockers':[],'source_blocker_count':0,'findings':findings,
    'remaining_execution_gates':['Separate root admission for exact current V4 numerical stage','Fresh numerical PASS with complete V3/V4 parity','Separate root admission and current full-real-graph parity plus complete original resource work','Exact current full-graph PASS and physical resource admission before scientific fit'],
    'runtime_parity_verified':False,'memory_feasibility_verified':False,'launch_authorized':False,
    'model_quality_or_manuscript_acceptance_judgment':False,
    'limitations':['Bounded parity is not exhaustive all-query proof','Arithmetic equality uses frozen tolerance, not bitwise CUDA identity','Historical environment not separately isolated; shared computational dependencies are byte-pinned and unchanged','Installed remote checkpoint.py bytes were not read; official version source was scoped and current deployed parity remains required'],
    'project_import_or_execution':False,'canonical_or_index_mutation':False,'candidate_edit':False})

for f in P.rglob('*.json'): json.loads(f.read_text())
for f in P.glob('*.py'): ast.parse(f.read_text(),filename=str(f))
assert sha(C/'MANIFEST.json')=='9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c'
assert sha(C/'SEAL.json')=='beffda0bbdbb3d71a232671349bb40b4ff1dec03b40c6ef81706b5ee287a19e3'
save('VERIFICATION.json',{'schema':'independent_V4_review_packet_verification_v1','UTC':UTC,
    'candidate_payloads':len(cm['files']),'predecessor_payloads':len(vm['files']),
    'candidate_and_predecessor_rehashed_without_execution':True,'candidate_manifest_and_seal_unchanged':True,
    'JSON_parsed':True,'review_script_AST_parsed':True,'project_Torch_or_GPU_execution':False})
files=sorted(f for f in P.rglob('*') if f.is_file() and f.name not in ['MANIFEST.json','SEAL.json'])
payload=[{'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
save('MANIFEST.json',{'schema':'independent_V4_checkpoint_review_manifest_v1','UTC':UTC,
    'status':'SEALED_SOURCE_ONLY_INDEPENDENT_REVIEW','files':payload,'payload_count':len(payload),
    'payload_bytes':sum(x['bytes'] for x in payload),'candidate_manifest_sha256':sha(C/'MANIFEST.json')})
for x in payload:
    f=P/x['path'];assert sha(f)==x['sha256'] and f.stat().st_size==x['bytes']
save('SEAL.json',{'schema':'independent_V4_checkpoint_review_seal_v1','UTC':UTC,
    'manifest_sha256':sha(P/'MANIFEST.json'),'manifest_bytes':(P/'MANIFEST.json').stat().st_size,
    'payload_count':len(payload),'payload_bytes':sum(x['bytes'] for x in payload),
    'verdict':'NO_IDENTIFIED_MATERIAL_SOURCE_BLOCKER_IN_SCOPED_CHANGE','project_or_GPU_execution':False,
    'report_sha256':sha(P/'REPORT.md'),'review_sha256':sha(P/'REVIEW.json')})
print(json.dumps({'manifest_sha256':sha(P/'MANIFEST.json'),'seal_sha256':sha(P/'SEAL.json'),
    'payload_count':len(payload),'payload_bytes':sum(x['bytes'] for x in payload)},indent=2))
