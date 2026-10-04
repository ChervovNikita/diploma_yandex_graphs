"""Independent local source/AST/custody verification; never execute target code."""
from pathlib import Path
import ast
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CANDIDATE = PHASE / 'pubmed_native_only_continuation_control_source_20261004_v2'
PREDECESSOR = PHASE / 'pubmed_native_only_continuation_control_source_20261004_v1'
PIN = '456a22a28420268e9ae47fd49f6ac322e36856b40a313340da4b31bde954ce7e'
SUPERVISOR_PIN = '7cd4432523e5227575fb58934b4cbc9762a2a47e503904d0cc529bcc1233385d'
seen = {}
hash_checks = []
ast_checks = []


def read(p):
    assert not p.is_symlink() and p.suffix in ('.py','.json','.md','.diff','.patch'), p
    raw = p.read_bytes()
    seen[str(p.relative_to(PHASE))] = dict(path=str(p.relative_to(PHASE)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return raw


def digest(p):
    return hashlib.sha256(read(p)).hexdigest()


def js(p):
    return json.loads(read(p))


def dump(n):
    return ast.dump(n, include_attributes=False)


def ast_sha(n):
    return hashlib.sha256(dump(n).encode()).hexdigest()


def tree(p):
    return ast.parse(read(p).decode())


def function(t, name):
    rows=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(rows)==1, name
    return rows[0]


def normalized(fn):
    fn=copy.deepcopy(fn);fn.name='normalized';return dump(fn)


def check(name, value, detail=None):
    ast_checks.append(dict(check=name,passed=bool(value),detail=detail))
    assert value, name


def pin(p,row):
    raw=read(p);ok=len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    hash_checks.append(dict(path=str(p.relative_to(PHASE)),expected_bytes=row['bytes'],expected_sha256=row['sha256'],passed=ok))
    assert ok,p


def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


manifest=js(CANDIDATE/'MANIFEST.json');seal=js(CANDIDATE/'SEAL.json')
check('exact requested candidate and seal',digest(CANDIDATE/'MANIFEST.json')==PIN and seal['manifest_sha256']==PIN and digest(CANDIDATE/'supervise.py')==SUPERVISOR_PIN and seal['successor_supervisor_sha256']==SUPERVISOR_PIN)
for row in manifest['files']:pin(CANDIDATE/row['path'],row)
inputs=js(CANDIDATE/'INPUTS.json')
for row in inputs['files']:pin(PHASE/row['path'],row)
preserved=js(CANDIDATE/'PREDECESSOR_PRESERVATION.json')
for row in preserved['snapshot_files']:pin(PHASE/row['path'],row)
binding=js(CANDIDATE/'SOURCE_BINDING.json');old_binding=js(PREDECESSOR/'SOURCE_BINDING.json')
for row in binding['external_source_pins']+binding['physical_helper_source_provenance_only']:pin(PHASE/row['path'],row)
check('all 14 active external source pins preserved',binding['external_source_pins']==old_binding['external_source_pins'] and len(binding['external_source_pins'])==14)
check('native prerequisite and teacher descriptors preserved',binding['prerequisite_receipts']==old_binding['prerequisite_receipts'] and binding['scientific_state_files_allowed'] is False)
old_pass=js(PHASE/'pubmed_native_only_continuation_control_fresh_source_review_20261004_v1/REVIEW.json')
corrective=js(PHASE/'pubmed_native_only_continuation_control_fresh_source_review_20261004_v2/REVIEW.json')
check('original mistaken PASS and corrective BLOCKED remain exact',old_pass['status']=='PASS' and corrective['status']=='BLOCKED' and corrective['candidate_manifest_sha256']==digest(PREDECESSOR/'MANIFEST.json') and [f['id'] for f in corrective['blocking_findings']]==['F06'] and digest(PHASE/binding['corrective_independent_predecessor_review']['path'])==binding['corrective_independent_predecessor_review']['sha256'])
unchanged=['common.py','native_continuation_control.py','step_observer.py','PLAN.json','ROOT_RELEASE_TEMPLATE.json','INDEPENDENT_SOURCE_REVIEW_TEMPLATE.json','PRESERVED_SHARED4_DIAGNOSTIC_SUMMARY.json','REPEATABILITY_ASSESSMENT.json']
for name in unchanged:check('byte-identical predecessor payload '+name,read(CANDIDATE/name)==read(PREDECESSOR/name))
old_tree=tree(PREDECESSOR/'supervise.py');new_tree=tree(CANDIDATE/'supervise.py')
census_tree=tree(PHASE/'graph_count_conditioned_train_support_census_preparation_20261004_v3/supervise.py')
for fn in [n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name!='main']:
    check('unchanged supervisor helper '+fn.name,dump(fn)==dump(function(new_tree,fn.name)))
check('collection source bytes preserved',ast.get_source_segment(read(PREDECESSOR/'supervise.py').decode(),function(old_tree,'collect'))==ast.get_source_segment(read(CANDIDATE/'supervise.py').decode(),function(new_tree,'collect')))
old_main=function(old_tree,'main');new_main=function(new_tree,'main');census_main=function(census_tree,'main')
handler=next(n for n in new_main.body if isinstance(n,ast.FunctionDef) and n.name=='interrupted')
census_handler=next(n for n in census_main.body if isinstance(n,ast.FunctionDef) and n.name=='interrupted')
check('handler exactly matches reviewed census-v3 AST',dump(handler)==dump(census_handler))
expected_nonlocal=ast.parse('nonlocal stop').body[0]
expected_stop=ast.parse('stop = stop or "SUPERVISOR_SIGNAL"').body[0]
expected_drain=ast.parse('if received:\n stop = stop or "SUPERVISOR_SIGNAL"').body[0]
expected_receipt=ast.parse('dict(received_supervisor_signals=list(received))').body[0].value.keywords[0]
check('immediate handler stop and received append',len(handler.body)==3 and dump(handler.body[0])==dump(expected_nonlocal) and dump(handler.body[1])==dump(ast.parse('received.append(number)').body[0]) and dump(handler.body[2])==dump(expected_stop))
main_try=next(n for n in new_main.body if isinstance(n,ast.Try))
drain_index=next(i for i,n in enumerate(main_try.finalbody) if dump(n)==dump(expected_drain))
restore=main_try.finalbody[drain_index-1]
physical=main_try.finalbody[drain_index+1]
check('drain strictly after handler restoration before physical construction',isinstance(restore,ast.For) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='signal' for n in ast.walk(restore)) and isinstance(physical,ast.Assign) and isinstance(physical.targets[0],ast.Name) and physical.targets[0].id=='physical')
signal_keyword=next(k for k in physical.value.keywords if k.arg=='received_supervisor_signals')
check('recorded signal list is copied into physical',dump(signal_keyword)==dump(expected_receipt))
check('drain and receipt match census-v3',any(dump(n)==dump(expected_drain) for n in ast.walk(census_main)) and any(isinstance(n,ast.keyword) and dump(n)==dump(expected_receipt) for n in ast.walk(census_main)))
stripped=copy.deepcopy(new_tree)
removed={'nonlocal':0,'handler_stop':0,'drain':0,'physical_signal_keyword':0}
stripped_main=function(stripped,'main')
stripped_handler=next(n for n in stripped_main.body if isinstance(n,ast.FunctionDef) and n.name=='interrupted')
for key,expected in [('nonlocal',expected_nonlocal),('handler_stop',expected_stop)]:
    for i,n in enumerate(stripped_handler.body):
        if dump(n)==dump(expected):stripped_handler.body.pop(i);removed[key]+=1;break
stripped_try=next(n for n in stripped_main.body if isinstance(n,ast.Try))
for i,n in enumerate(stripped_try.finalbody):
    if dump(n)==dump(expected_drain):stripped_try.finalbody.pop(i);removed['drain']+=1;break
stripped_physical=next(n for n in stripped_try.finalbody if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='physical')
for i,k in enumerate(stripped_physical.value.keywords):
    if k.arg=='received_supervisor_signals':stripped_physical.value.keywords.pop(i);removed['physical_signal_keyword']+=1;break
check('only four documented insertions removed',all(v==1 for v in removed.values()),removed)
check('entire module AST restored to predecessor',dump(stripped)==dump(old_tree),dict(predecessor_AST_sha256=ast_sha(old_tree),stripped_successor_AST_sha256=ast_sha(stripped)))
gate_test=ast.parse('physical_complete and reaped and stop is None and process.returncode==0',mode='eval').body
check('physical success rejects any stop before collect',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and n.args and dump(n.args[0])==dump(gate_test) for n in ast.walk(new_main)))
check('terminal inherits physical receipt',any(isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='terminal' and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='dict' and n.value.args and isinstance(n.value.args[0],ast.Name) and n.value.args[0].id=='physical' for n in ast.walk(new_main)))

common=tree(CANDIDATE/'common.py');worker=tree(CANDIDATE/'native_continuation_control.py');plan=js(CANDIDATE/'PLAN.json');template=js(CANDIDATE/'ROOT_RELEASE_TEMPLATE.json')
for f in ['common.py','native_continuation_control.py','step_observer.py','supervise.py']:tree(CANDIDATE/f)
work=function(common,'native_work')
before=next(n for n in work.body if isinstance(n,ast.FunctionDef) and n.name=='before')
check('pre-hook refuses update 253 before increment',len(before.body)==2 and 'Adam_started' in ast.get_source_segment(read(CANDIDATE/'common.py').decode(),before) and isinstance(before.body[0],ast.Expr) and isinstance(before.body[0].value,ast.Call) and isinstance(before.body[0].value.args[0],ast.Compare) and isinstance(before.body[0].value.args[0].ops[0],ast.Lt) and isinstance(before.body[0].value.args[0].comparators[0],ast.Name) and before.body[0].value.args[0].comparators[0].id=='maximum_updates')
work_calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='native_work']
check('all warm and continuation hooks bind 252',len(work_calls)==2 and all(ast.literal_eval(n.args[-1])==252 for n in work_calls))
check('exact declared 7 epochs and 252 updates',plan['stages']['native_continuation_control']['invocation']['engineering_Adam_updates']==252 and plan['stages']['native_continuation_control']['invocation']['warm_Adam_updates']==180 and plan['stages']['native_continuation_control']['invocation']['continuation_Adam_updates']==72 and plan['stages']['native_continuation_control']['invocation']['full_batches_per_epoch']==36)
observer=PHASE/plan['observer_source_path']
pin(observer,next(r for r in binding['external_source_pins'] if r['path']==plan['observer_source_path']))
check('step observer byte-identical preserved shared4-v3',read(CANDIDATE/'step_observer.py')==read(PHASE/'pubmed_shared4_owned_continuation_diagnostic_source_20261004_v3/step_observer.py'))
bodies=tree(PHASE/'pubmed_heart_native_numerical_qualification_source_20261004_v2/native_bodies.py')
author=tree(PHASE/'pubmed_heart_available_inspector_native_adapter_20261004_v1/public_author_code/HeaRT/benchmarking/HeaRT_small/main_ncn_CoraCiteseerPubmed.py')
check('reference/candidate/public-author native train normalized AST',normalized(function(bodies,'reference_ncnc_train'))==normalized(function(bodies,'candidate_ncnc_train'))==normalized(function(author,'train')))
compare=function(tree(PHASE/plan['comparator_source_path']),'compare')
check('fixed original comparator AST pin',ast_sha(compare)==js(CANDIDATE/'SOURCE_EQUIVALENCE.json')['fixed_original_compare_AST_sha256'])
feature=plan['input_authority']['feature_equivalence_receipt'];check('local scalar feature authority pin',digest(PHASE/feature['path'])==feature['sha256'])
prerequisite_availability=[dict(name=k,**v,locally_present=(PHASE/v['path']).exists(),independently_server_authenticated=False) for k,v in binding['prerequisite_receipts'].items()]
check('disabled source-only template',template.get('status')!='APPROVED' and template.get('authorized_stages')==[] and template.get('scientific_fit_admitted') is False and template.get('state_donor_allowed') is False)
save('INPUTS.json',dict(files=list(seen.values()),source_only=True,numerical_modules_imported=False,target_sources_imported_compiled_executed=False,server_access=False,raw_feature_train_or_state_arrays_read=False))
save('HASH_VERIFICATION.json',dict(status='PASS',candidate_manifest_sha256=PIN,successor_supervisor_sha256=SUPERVISOR_PIN,candidate_payload_count=len(manifest['files']),candidate_payload_bytes=sum(r['bytes'] for r in manifest['files']),input_rows=len(inputs['files']),input_bytes=sum(r['bytes'] for r in inputs['files']),preservation_rows=len(preserved['snapshot_files']),checks=hash_checks,missing_local_native_prerequisite_receipts=prerequisite_availability))
save('AST_VERIFICATION.json',dict(status='PASS',checks=ast_checks,check_count=len(ast_checks),blocking_findings=[],minimal_delta_removed=removed,entire_supervisor_AST_sha256=ast_sha(new_tree),stripped_successor_AST_sha256=ast_sha(stripped),predecessor_AST_sha256=ast_sha(old_tree),candidate_source_executed=False))
print(json.dumps(dict(status='PASS_LOCAL_SOURCE_VERIFICATION',hash_checks=len(hash_checks),AST_checks=len(ast_checks),candidate_payload_count=len(manifest['files']),candidate_payload_bytes=sum(r['bytes'] for r in manifest['files']),input_rows=len(inputs['files']),input_bytes=sum(r['bytes'] for r in inputs['files']),preservation_rows=len(preserved['snapshot_files']),prerequisite_availability=prerequisite_availability),indent=2))
