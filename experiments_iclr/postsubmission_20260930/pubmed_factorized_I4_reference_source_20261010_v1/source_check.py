"""Pure stdlib source checks; no provider/model/checkpoint/array import or read."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
SERVER_PHASE='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/'
PAYLOAD=('BINDINGS.json','ROSTER.json','DATA_AND_RUNTIME.json','RELEASE_TEMPLATE_DISABLED.json','ROOT_REVIEW_TEMPLATE_DISABLED.json','OWNER_REVIEW_TEMPLATE_DISABLED.json','PROTOCOL.json','COST_FORECAST.json','README.md','common.py','run.py','owner.py','render.py','source_check.py','STATIC_CHECK.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parsed={}
    for name in PAYLOAD:
        if name=='STATIC_CHECK.json':continue
        path=HERE/name
        if path.suffix=='.py':parsed[name]=ast.parse(path.read_text());compile(path.read_text(),str(path),'exec')
        elif path.suffix=='.json':json.loads(path.read_text())
    binding=json.loads((HERE/'BINDINGS.json').read_text());roster=json.loads((HERE/'ROSTER.json').read_text())
    for key in ('M1_source_manifest','reference_source_manifest','M1_owner_source','history_plan','history_reuse','history_seal'):
        row=binding[key];path=PHASE/row['path'].removeprefix(SERVER_PHASE)
        if not row['path'].startswith(SERVER_PHASE) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Dependency source binding changed: '+key)
    for key in ('M1_source_manifest','reference_source_manifest'):
        row=binding[key];path=PHASE/row['path'].removeprefix(SERVER_PHASE)
        for file in json.loads(path.read_text())['files']:
            payload=path.parent/file['path']
            if sha(payload)!=file['sha256'] or payload.stat().st_size!=file['bytes']:raise ValueError('Original source payload changed')
    plan=json.loads((PHASE/binding['history_plan']['path'].removeprefix(SERVER_PHASE)).read_text())
    if [{k:r[k] for k in r if k!='body_index'} for r in roster]!=plan['roster']:raise ValueError('History exact12 roster changed')
    if len(roster)!=12 or len({r['record_id'] for r in roster})!=12 or sum(r['body_index']>0 for r in roster)!=9:raise ValueError('Fixed9+3 positions required')
    if any(r['enabled'] is not False or r['initialization_seed']!=r['seed_block']+1009*r['body_index'] or r['factual_dropout_seeds']!=[r['initialization_seed']+300001] for r in roster):raise ValueError('Frozen seeds required')
    template=json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
    for key in ('enabled','root_authorized','source_review_approved','complete_roster_frozen','complete_input_custody_verified','external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','TEST_access','automatic_retry','resume','fresh12_fallback','confirmation_claim','paper_score_recalculation'):
        if template[key] is not False:raise ValueError('Inactive closed template required')
    if any(path.name in ('ROOT_SOURCE_REVIEW.json','OWNER_REVIEW.json','READINESS.json','ROOT_ADOPTION.json') for path in HERE.iterdir()):raise ValueError('Source check packet must contain no actual approvals')
    if any(name.endswith(('_OWNER_PLAN.json',)) for name in PAYLOAD):raise ValueError('Actual plans excluded from source')
    imports=[]
    for name,tree in parsed.items():
        for node in tree.body:
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
    if any(n.split('.')[0] in ('numpy','torch','scipy','torch_geometric') for n in imports):raise ValueError('No numerical import at module initialization')
    run=(HERE/'run.py').read_text();owner=(HERE/'owner.py').read_text();common=(HERE/'common.py').read_text();render=(HERE/'render.py').read_text()
    for token in ("factory.fresh_single('single_native'",'engine.fit_body(',"member_context(s.bodies[0],0)",'bank.softmax(-1).mean(0)','s=None;gc.collect()',"closed_stage('science'", "closed_stage('assembly'"):
        if token not in run:raise ValueError('Required unchanged/native/new interface token absent: '+token)
    if 'assembled_I4(' in run or 'def own_one(' in owner or 'old.own_one(row,plan)' not in owner:raise ValueError('Old fit/owner duplication or forbidden factor-free assembly')
    if "if mode=='qualification':closed_stage('admission'" not in common or "if stage=='qualification':closed_stage('admission'" not in render:raise ValueError('Anchor admission must precede every new update')
    for token in ('complete_sha256','plan_sha256','all_records_directly_waited','quality_fields_opened','owned_CUDA_absence_verified'):
        if token not in common:raise ValueError('Full custody closure gate absent')
    result=dict(schema='PubMed-factorized-I4-source-static-check-v1',complete=True,pure_stdlib=True,compiled_modules=sorted(parsed),JSON_parse=True,original_source_bindings_and_payloads_verified=True,history_exact12_roster_verified=True,nine_new_fits_three_reused_positions=True,inactive_templates_verified=True,unchanged_owner_call_verified=True,new_four_private_M1_row0_interface_present=True,anchor_before_every_new_update=True,full_raw_wait_and_complete_hash_closure_present=True,
                source_only=True,numerical_import_or_execution=False,model_checkpoint_or_raw_array_read=False,qualification_execution=False,remote_operation=False,limitations='Static source/interface/authority checks only. Actual providers, GPU memory and four-body restore/replay remain root-owned engineering gates.')
    (HERE/'STATIC_CHECK.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
