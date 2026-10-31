"""Stdlib-only, source-bound engineering blueprint; no numerical execution API."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

MASTER_HASH = '081b1358d83468c8062844288f28c9f741984176d441466df8c7daded82a57ef'
CAP_HELPER_SHA = 'effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'
ORIGINAL_FIVE = {'romanempire', 'amazonratings', 'minesweeper', 'questions', 'tolokers'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def confined(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Project-relative authority required')
    value = root/path
    if value.is_symlink() or not value.resolve().is_relative_to(root):
        raise ValueError('Confined nonsymlink authority required')
    return value


def receipt(root, path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def role_paths(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        if {'path','sha256','bytes'} <= set(value):
            return {value['path']}
        return set().union(*(role_paths(v) for v in value.values())) if value else set()
    return set()


def read_custody(root, study):
    for row in study['source']['files']:
        if Path(row['path']).name == 'CUSTODY.json':
            path = confined(root,row['path'])
            if receipt(root,path) != row:
                raise ValueError('Custody metadata changed')
            value = json.loads(path.read_text())
            if value.get('schema') == 'node-provider-role-custody-v1' and value.get('dataset_id') == study['dataset']:
                return value, row
    raise ValueError('Actual prepared node custody metadata required')


def shape_and_anchor(family, run):
    c = run['config']
    if family == 'molecular':
        return (c['model']['hidden'], c['model']['layers'], c['model']['dropout']), c['optimizer']['learning_rate'] == .001 and c['model']['dropout'] == .5
    if family == 'link':
        return (c['encoder']['hidden'], c['encoder']['layers'], c['encoder']['dropout']), c['encoder']['hidden'] == 64 and c['optimizer']['encoder_lr'] == .0082 and c['optimizer']['predictor_lr'] == .0037
    return (c['width'],c['depth'],c['dropout']), c['learning_rate'] == .001 and c['dropout'] == .4


def representatives(family, rows):
    shapes = {shape_and_anchor(family,r)[0] for r in rows}
    selected = []
    for shape in sorted(shapes):
        values = [r for r in rows if shape_and_anchor(family,r)[0] == shape]
        anchors = [r for r in values if shape_and_anchor(family,r)[1]]
        frontier = not any(all(a >= b for a,b in zip(other,shape)) and other != shape for other in shapes)
        if anchors or frontier:
            selected.append((min(anchors or values,key=lambda r:(r['seed'],r['run_id'])),bool(anchors),frontier))
    return selected


def prepare(root, plan_path):
    plan = json.loads(plan_path.read_text())
    if digest(plan) != MASTER_HASH or plan['scientific_launch_enabled'] is not False:
        raise ValueError('Exact disabled v3 master required')
    cases = []
    for entry in plan['studies']:
        path = confined(root,entry['path']);raw=path.read_bytes()
        study = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        if digest(study) != entry['spec_hash']:
            raise ValueError('Bound v3 study changed')
        identifier = ''.join(c for c in study['dataset'].lower() if c.isalnum())
        if identifier in ORIGINAL_FIVE:
            raise ValueError('Original-paper datasets excluded')
        runs = {r['run_id']:r for r in study['runs']}
        author = [r for r in study['runs'] if r['kind'] == 'author_cli_group']
        task = study['data'].get('task')
        is_pt = any('retained_pt_routes.py' in r.get('adapter','') for r in runs.values())
        if not author and task not in {'graph_binary_classification','link_prediction'} and not is_pt:
            continue
        study_binding = receipt(root,path)
        prohibited = set().union(*(role_paths(v) for k,v in study['data']['roles'].items() if k != 'train'))
        evidence=study['data'].get('evidence',{})
        if isinstance(evidence,dict):
            prohibited.update(role_paths(evidence.get('endpoint_inputs',{})))
        common = {'study_id':study['study_id'],'dataset':study['dataset'],'source_spec_hash':entry['spec_hash'],
                  'study_binding':study_binding,'source_files':[r for r in study['source']['files'] if Path(r['path']).suffix == '.py'],
                  'prohibited_original_role_paths':sorted(prohibited),'scientific_fits':0,'scientific_optimizer_updates':0,
                  'CUDA_execution_authorized':False,'source_exact_recipe_ready':False,'scientific_launch_ready':False,
                  'all_fit_ETA':None,'numerical_extension_implementation_and_root_review_required':True}
        if author:
            for run in author:
                argv=run['cli']['argv'];native_dataset=argv[argv.index('--dataset')+1]
                dataset = {'cs':'coauthor-cs','physics':'coauthor-physics'}.get(native_dataset,native_dataset)
                # The prepared role is a declared resource proxy. Native author
                # class-balanced60% custody is not inferred or reconstructed.
                cpath = confined(root,'training_pipeline_v2_20261012/prepared_data/'+dataset+'/CUSTODY.json')
                custody = json.loads(cpath.read_text())
                rows = custody['splits'];maximum=max(r['role_counts']['train'] for r in rows)
                chosen=min((r for r in rows if r['role_counts']['train']==maximum),key=lambda r:r['split_id'])
                case = dict(common,family='author_direct_builder',declared_author_run=run,prepared_custody_binding=receipt(root,cpath),
                            prepared_proxy_mask=chosen['split_id'],prepared_proxy_TRAIN_rows=maximum,
                            builder_entrypoint='pinned mymodels.PolyFormer(dataset_interface,args); pinned utils.get_base(mono_or_cheb,K,X,A)',
                            parameter_recipe='literal declared CLI argv plus static literal parser defaults; one source Adam group per registered parameter',
                            input_authority='prepared target-free GRAPH plus existing prepared TRAIN only; never native raw provider labels or DataLoader',
                            scope='first_repeat_random_builder_and_prepared_TRAIN_proxy_only',
                            native_author_TRAIN_custody_qualified=False,native_author_CLI_process_executed=False,
                            native_global_RNG_repeat_trajectory_or_patience_qualified=False,
                            selected_checkpoint_serializer_qualified=False,
                            snapshot_proxy='engineering-only model/Adam/RNG storage; original author has no saved selected checkpoint')
                case['prohibited_original_role_paths']=sorted(prohibited | role_paths(run['cli']['raw_files'][0])
                    | set().union(*(role_paths(v) for v in run['cli']['raw_files'])))
                case['case_id']=digest({'study':entry['spec_hash'],'author':run['run_id'],'mode':case['scope']})[:24]
                cases.append(case)
            continue
        family = 'pt_random_direct_builder' if is_pt else 'molecular' if task == 'graph_binary_classification' else 'link'
        custody = None
        if is_pt:
            custody,custody_binding=read_custody(root,study)
            counts={r['split_id']:r['role_counts']['train'] for r in custody['splits']}
        for selection in [r for r in runs.values() if r['kind']=='selection']:
            candidates=[runs[c['run_ids'][0]] for c in selection['config']['configurations']]
            inputs_cohort=None
            if is_pt:
                comparative=[r for r in runs.values() if r['kind']=='model_fit' and r['config'].get('selected_config_from')==selection['run_id']]
                masks=sorted({r['split'] for r in comparative});maximum=max(counts[m] for m in masks)
                mask=min(m for m in masks if counts[m]==maximum)
                inputs_cohort=min((r for r in comparative if r['split']==mask),key=lambda r:(r['body'],r['seed'],r['run_id']))
            for run,anchor,frontier in representatives(family,candidates):
                case=dict(common,family=family,selection_run_id=selection['run_id'],declared_source_run=run,
                          source_anchor=anchor,shape_frontier=frontier,shape=list(shape_and_anchor(family,run)[0]),
                          coverage='source_shape_envelope_and_random_timing_proxy_only;lower_points_masks_seeds_and_selected_states_unqualified',
                          warm_iterations_proposed=3,case_timeout_seconds_proposed=600,
                          allocator_cap='min(fresh_free_bytes-4GiB,floor(0.4*total_bytes)); current-child normal Torch allocator only',
                          optimizer_policy='deny concrete and functional actions before factory; manually fresh zero Adam/AdamW moments; exact parameter/moment hashes',
                          snapshot_policy='engineering-only random full-root state proxy; never selected, never registry ancestor, remove temporary binary',
                          failure_policy='fresh PID/start/PGID before numerics; persist external usage/free-total on deferral; retain failures/costs; no retry')
                if family == 'molecular':
                    case.update(builder_entrypoint=run['adapter'],actual_TRAIN_cost=run['config']['molecular_cost'],
                                input_authority='actual immutable TRAIN graph+label pack only',
                                unused_SELECT_interface='fully synthetic one-atom zero-category molecule; no edges; synthetic positive graph identity2**62 and zero target; never forwarded',
                                measure_unit='one complete source TRAIN epoch including actual tail, cold plus three warm; all returned losses ignored')
                elif family == 'link':
                    case.update(builder_entrypoint=run['adapter'],input_authority='actual immutable GRAPH features/raw bidirectional TRAIN graph plus TRAIN positive edge records only',
                                unused_SELECT_interface='one synthetic positive pair plus50 distinct synthetic negative pairs, public GRAPH endpoints chosen outside TRAIN canonical pairs; never forwarded',
                                measure_unit='one complete native TRAIN epoch; exact sampler/maskinput/support rebuilds/floor batches and dropped tail, cold plus three warm',
                                train_minibatches_per_epoch=run['config']['train_minibatches_per_epoch'],
                                serving_or_Hits50_endpoint_qualified=False)
                else:
                    c=run['config'];case.update(prepared_custody_binding=custody_binding,source_input_cohort_run_id=inputs_cohort['run_id'],
                        resource_GRAPH_TRAIN_inputs={k:inputs_cohort['config']['node_inputs'][k] for k in ('graph','train')},
                        representative_mask=mask,actual_TRAIN_rows=maximum,
                        builder_entrypoint='pinned retained_pt compile_source/native.build_arm and pt.cache_stem/pt.build_arm; explicit direct Bank assembly',
                        source_stage=c['pt_stage'],pt_arm=c['pt_arm'],
                        source_prototype='fresh random exact native P1 (or F1 for continued_f1), parameter hashes and manual zero moments; zero updates; not paid/selected',
                        random_H0='full public GRAPH undropped cache_stem from that random prototype; hash/cache bytes/preparation calls recorded separately',
                        original_registry_or_selected_parent_called=False,fake_paid_full_source_receipt_created=False,
                        fake_scientific_ancestor_created=False,scientific_parent_receipt=None,
                        actual_paid_selected_parent_constructor_restore_or_H0_cost_qualified=False,
                        named_controls='source P1/F1/S4/original; head_only,mixed_single,private4,middle_state_mean,bias_coordinate_single; NEW separable4/continued_p1/continued_f1 remain distinct',
                        I4_qualification='one own body resource prototype; never four-view bank equivalence or paid ancestor qualification',
                        measure_unit='full GRAPH TRAIN source objective/regularizer FWD/backward; source operator counts, cold plus three warm; no nominal updates persisted')
                case['case_id']=digest({'study':entry['spec_hash'],'run':run,'cohort':case.get('source_input_cohort_run_id'),'mode':family})[:24]
                cases.append(case)
    return {'schema':'practical-v3-resource-extension-engineering-blueprint-v1','master_hash':MASTER_HASH,
            'master_binding':receipt(root,plan_path),'preparer_binding':receipt(root,Path(__file__)),
            'reviewed_cap_denial_clone_helper_sha256':CAP_HELPER_SHA,'cases':cases,
            'status':'BLUEPRINT_ONLY_NUMERICAL_IMPLEMENTATION_AND_ROOT_REVIEW_PENDING','numerical_execution_API_present':False,
            'CUDA_cases_executed':0,'scientific_fits':0,'scores_emitted':0,'role_tensor_decoding':False,
            'scientific_launch_enabled':False,'human_GO_required':True,'original_five_excluded':True,
            'all_fit_ETA':None,'full_native_author_custody_and_selected_paid_PT_ancestry_qualified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert subprocess.check_output(['hostname'],text=True).strip()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    root=Path(args.project_root).resolve();output=confined(root,args.output);output.mkdir(exist_ok=False)
    value=prepare(root,confined(root,'massive_practical_master_20261012_v3/PLAN.json'))
    (output/'BLUEPRINT.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    from collections import Counter
    print(json.dumps({'metadata_only':True,'cases':len(value['cases']),'by_family':dict(Counter(c['family'] for c in value['cases'])),
                      'CUDA_cases_executed':0,'scientific_fits':0,'scores_emitted':0}))


if __name__=='__main__':
    main()
