"""Audit complete metadata receipts; never open a tensor or label pack."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE=Path(__file__).resolve().parents[1]
ROOT=PHASE/'modern_teacher_execution_root_v1'
REMOTE='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def desc(path):return dict(path=REMOTE+str(path.relative_to(PHASE)),sha256=sha(path))
def require(condition,message):
    if not condition:raise ValueError(message)
def local(record):
    require(record['path'].startswith(REMOTE),'Unconfined descriptor')
    path=PHASE/record['path'][len(REMOTE):]
    require(path.resolve().is_relative_to(PHASE) and sha(path)==record['sha256'],'Local descriptor differs')
    return path

def main():
    rows=[]
    manifest=desc(PHASE/'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v1/MANIFEST.json')
    for backbone,graph in (('polyformer_mono','Squirrel'),('polynormer_r','Photo')):
        for family in ('single_author','gnnm_boundary_4','independent_author_4_same_width'):
            run='run05' if (backbone,family)==('polynormer_r','gnnm_boundary_4') else 'run04'
            identity=backbone+'__'+family+'__config0_seed17_'+run
            out=ROOT/identity;request_path=ROOT/(identity+'_REQUEST.json')
            request=read(request_path);admission=read(local(request['teacher_admission']))
            terminal=read(out/'TERMINAL.json');qual_path=out/'cell/QUALIFICATION_RECEIPT.json'
            external_path=out/'EXTERNAL_OPTIMIZER_AUDIT.json';qual=read(qual_path);external=read(external_path)
            cell=read(out/'cell/TEACHER_CELL_FREEZE.json');selection=read(out/'cell/teacher_selection.json')
            bound=read(ROOT/(identity+'_bound')/'TERMINAL.json')
            authorized=read(ROOT/(identity+'_supervisor')/'authorization.json')
            require(terminal['complete'] and not terminal['report_eligible'] and not terminal['final_labels_read'], 'Wrong terminal scope')
            require(terminal['request_sha256']==sha(request_path) and terminal['qualification_receipt_sha256']==sha(qual_path) and terminal['optimizer_audit_sha256']==sha(external_path), 'Terminal hashes differ')
            require(bound['complete'] and bound['child_exit_code']==0 and not bound['timed_out'] and bound['root_request_unchanged'], 'Whole supervisor did not close normally')
            require(authorized['child_exit_code']==0 and authorized['visible_gpu_count']==1 and authorized['gpu_uuid']=='GPU-44039938-fd82-41d2-fefd-de71514e2fac', 'Allocation differs')
            require(request['source_manifest']==manifest and request['deterministic'] is True and request['aps_score_backend']=='cpu_fixed_algorithm', 'Wrong source/backend')
            require(admission['environment']['deterministic_algorithms'] is True and admission['environment']['cublas_workspace_config']==':4096:8', 'Wrong deterministic environment')
            require(qual['environment']==admission['environment']==cell['environment'] and qual['implementation_sha256']==admission['implementation_sha256'], 'Runtime/source custody differs')
            require(qual['backbone']==backbone and qual['family']==family and cell['graph']==graph and not cell['report_eligible'] and selection['qualification_only'], 'Family/scope differs')
            parity=qual['parity']
            require((parity['rtol_logits'],parity['atol_logits'],parity['rtol_gradients'],parity['atol_gradients'])==(1e-5,1e-6,1e-4,1e-6), 'Tolerance changed')
            require(parity['native_vs_identity_boundary_passed'] and parity['shortened_fit_report_eligible'] is False and parity['selected_checkpoint_replay_max_absolute_logit_difference']==0, 'Parity/replay not established')
            stages=[False,True] if backbone=='polynormer_r' else [False]
            require([r['global_stage'] for r in parity['checks']]==stages, 'Incomplete stages')
            require(all(r['full_graph'] and r['all_graph_logits_and_active_gradients_finite'] and r['dormant_owner_removal_verified'] and r['max_absolute_logit_difference']==0 and r['common_body_gradient_tensors_compared']>0 and r['private_bias_gradient_rows_compared']==8 for r in parity['checks']), 'Incomplete full-graph checks')
            steps=4 if backbone=='polynormer_r' else 3
            require(len(external['steps'])==steps and external['selected_full_graph_logits_finite'], 'Incomplete training checks')
            members=1 if family=='single_author' else 4
            require(all(r['finite_gradient_tensors']>0 and len(r['member_gradient_energy'])==members and all(v>0 for v in r['member_gradient_energy']) for r in external['steps']), 'Member connectivity not established')
            require(len(external['optimizer_groups'])==1 and external['optimizer_groups'][0]['live_membership_exact'], 'Optimizer ownership not checked')
            require(len(external['restores'])==(1 if backbone=='polynormer_r' else 0), 'Adam transition differs')
            require(all(r['exact_optimizer_state_restored'] and r['step_counters_checked'] for r in external['restores']), 'Adam restore not exact')
            require(len(external['model_restores'])==(2 if backbone=='polynormer_r' else 1) and all(r['exact_model_state_restored'] for r in external['model_restores']), 'Model restore not exact')
            custody=external['exact_input_custody']
            require(custody['source_manifest']==manifest and all(custody[k]==admission[k] for k in ('role_freeze','source_labels','environment','backbone','family','implementation_sha256')), 'Supplement custody differs')
            roles=read(local(admission['role_freeze']))
            require(roles['seed']==17 and roles['source_split_index']==0 and custody['graph_input']==roles['graph_input'], 'Graph/role binding differs')
            label_manifest=read(PHASE/'coordinate_conformal_execution_root_v1/acquisition_run02'/graph/'LABEL_PACK_MANIFEST.json')
            pack=label_manifest['cells'][0]
            require(admission['source_labels']=={k:pack['source_labels'][k] for k in ('train','validation')} and admission['role_freeze']==pack['role_freeze'], 'Source labels/roles are not the acquired pairing')
            rows.append(dict(backbone=backbone,graph=graph,family=family,tested_seed=17,
                tested_source_split=0,request=desc(request_path),admission=request['teacher_admission'],
                qualification=desc(qual_path),external_optimizer_audit=desc(external_path),
                root_terminal=desc(out/'TERMINAL.json'),cell_freeze=desc(out/'cell/TEACHER_CELL_FREEZE.json'),
                parameters=selection['parameter_count'],stored_model_tensor_bytes=selection['model_tensor_bytes'],
                costs=read(out/'cell/COSTS.json'),whole_supervised_seconds=bound['whole_supervised_seconds'],
                report_eligible=False,full_fit_admitted=False,heldout_labels_read=False))
    result=dict(schema='gnnm-modern-six-full-graph-qualification-audit-v1',created_UTC=datetime.now(timezone.utc).isoformat(),
        all_six_metadata_gates_verified=True,rows=rows,numerically_tested_seed=17,
        label_or_tensor_arrays_opened_by_this_auditor=False,
        scope='Root audit of immutable supervised runtime/parity/optimizer receipts; no independent new forward replay and no scientific utility claim.',
        full_schedule_feasibility_or_scientific_admission=False,original_scores_changed=False)
    with (ROOT/'SIX_QUALIFICATION_AUDIT_v1.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(verified_rows=len(rows),scientific_fit_admission=False)))

if __name__=='__main__':main()
