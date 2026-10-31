"""Fresh standardized Coauthor coverage from frozen full-v2 source APIs.

Rebuild every procedure on all10 prepared masks, then declare the prospective nine
point policy. No source tensors, models, scores, fits or CUDA are imported.
"""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path

from adapters.polyformer_boundary import M4, M1, validate_config
from adapters.polyformer_endpoints import independent_contracts
from compile_native_matrix import bind_custody
from compile_polyformer_boundary_matrix import append_boundary, receipt, verify_receipt
from compile_polyformer_matrix import make_study, templates
from compile_polyformer_practical_matrix import COAUTHOR_POLICY, practical_study
from compile_molecular_matrix import make_plan
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate


def rebuild_coauthor(original, custody, data_dir, recipe, previous, root, runtime_prefix, files, policy):
    if policy != COAUTHOR_POLICY or original['dataset'] not in policy['datasets']:
        raise ValueError('Exact prospective Coauthor policy and full-v2 dataset required')
    if original['stage'] != 'development' or custody['dataset_id'] != original['dataset']:
        raise ValueError('Existing development custody and matching source dataset required')
    clone = copy.deepcopy(custody)
    clone['splits'] = sorted(clone['splits'], key=lambda row:row['split_id'])
    if ([r['split_id'] for r in clone['splits']] != policy['comparative_splits']
            or any(r['evaluate_purpose'] != 'development' for r in clone['splits'])):
        raise ValueError('All10 prepared development comparative masks required')
    source = copy.deepcopy(original['source'])
    old_prefix = original['runs'][0]['adapter'].split('/adapters/')[0]
    source['files'] = [r for r in source['files'] if not r['path'].startswith(old_prefix + '/')] + files
    source['files'] = list({r['path']:r for r in source['files']}.values())
    base, counts, banks, domains = make_study(root, clone, data_dir, recipe, source,
        original['resources']['profile'], 'native_fixed', runtime_prefix)
    full, fc, fb, fd = append_boundary(base, {'counts':counts,'served_banks':banks,'search_domains':domains}, runtime_prefix, [])
    value, total, served, declared = practical_study(full,
        {'counts':fc,'served_banks':fb,'search_domains':fd}, recipe, policy, runtime_prefix, files)
    value['source']['unrestricted_v2_ancestry'] = {'study_id':original['study_id'], 'spec_hash':fingerprint(original),
        'unrestricted_packet_modified':False, 'all10_templates_reconstructed_from_unchanged_source_compiler_APIs':True}
    value['source']['coauthor_coverage_policy'] = {'track':'standardized_development', 'search_mask':0, 'comparative_masks':policy['comparative_splits'],
        'author_score_protocol_equality':False, 'author_track_packets_modified':False,
        'native_source_shapes_and_grouped_LR_anchor_retained':True, 'all36_procedures_and55_served_methods_retained':True,
        'source_recipe_id':recipe['recipe_id'], 'source_native_point':recipe['native_point'],
        'old_v2_comparative_masks':original['data']['evidence']['split_roster'],
        'all_prepared_masks_generated_from_source_APIs_not_selected_run_rows':True}
    value['hypothesis'] += ' Prospective Coauthor coverage on search mask0 and all10 prepared comparative masks; standardized development, author-score protocol equality not claimed.'
    expected = {(mask,s) for mask in policy['comparative_splits'] for s in policy['comparative_seeds']}
    exported = [r for r in value['runs'] if r['kind']=='ensemble_assembly']
    for comparison in value['comparisons']:
        if comparison['original_GNNM_reference'] != M4:
            raise ValueError('Actual source-boundary original GNNM reference required')
        for method in comparison['references'] + comparison['candidates']:
            if {(r['split'],r['seed']) for r in exported if r['method']==method} != expected:
                raise ValueError('Every comparison must retain its complete all10-mask/three-seed cohort')
    if len({r['method'] for r in exported}) != 55 or len(declared) != 36:
        raise ValueError('Every implemented source mechanism and reference required')
    if total['optimizer_starts_by_stage'] != {'search':648,'comparative':2790}:
        raise ValueError('Exact finite Coauthor search/comparison counts required')
    validate(value)
    return value, total, served, declared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','v2-packet','runtime-prefix','policy','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();verify_allocation();root=Path(args.project_root).resolve()
    old=confined(root,args.v2_packet);runtime=confined(root,args.runtime_prefix);policy_path=confined(root,args.policy)
    policy=json.loads(policy_path.read_text())
    if policy != COAUTHOR_POLICY:raise ValueError('Exact frozen Coauthor policy required')
    old_plan=json.loads((old/'matrix/PLAN.json').read_text());old_summary=json.loads((old/'matrix/SUMMARY.json').read_text())
    if fingerprint(old_plan) != old_summary['plan_hash'] or old_plan['scientific_launch_enabled'] or old_summary['GO_created']:
        raise ValueError('Full-v2 source packet must remain disabled and immutable')
    catalogue=json.loads((runtime/'docs/MATRIX_AXES_DRAFT.json').read_text())['baseline_recipe_catalogue']['rows']
    files=[receipt(root,p.relative_to(root)) for p in sorted(runtime.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md')]
    files += [receipt(root,policy_path.relative_to(root))] + [receipt(root,(old/n).relative_to(root)) for n in ('matrix/PLAN.json','matrix/SUMMARY.json')]
    prepared=[];source_metadata=[]
    for row in old_plan['studies']:
        path=confined(root,row['path']);original=json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(original) != row['spec_hash']:raise ValueError('Exact source full-v2 study changed')
        if original['dataset'] not in policy['datasets']:continue
        for binding in original['source']['files']:verify_receipt(root,binding)
        dataset=original['dataset'];custody_rows=[r for r in original['source']['files'] if r['path'].endswith('/'+dataset+'/CUSTODY.json')]
        if len(custody_rows)!=1:raise ValueError('Exactly one existing dataset custody required')
        custody,data_dir,bindings=bind_custody(root,confined(root,custody_rows[0]['path']))
        recipe=next(r for r in catalogue if r['recipe_id']=='polyformer_mono_'+dataset)
        previous=next(r for r in old_summary['per_study'] if r['study_id']==original['study_id'])
        prepared.append(rebuild_coauthor(original,custody,data_dir,recipe,previous,root,args.runtime_prefix,
            files+bindings+[receipt(root,path.relative_to(root))],policy))
        source_metadata.append({'dataset':dataset,'source_study':receipt(root,path.relative_to(root)),
            'source_spec_hash':row['spec_hash'],'custody':custody_rows[0], 'model_graph':custody['model_graph'],
            'role_counts_by_mask':{str(r['split_id']):r['role_counts'] for r in custody['splits']},
            'native_point':recipe['native_point'],'operator':recipe['operator'],'locks':recipe['locks']})
    if {s['dataset'] for s,*_ in prepared} != set(policy['datasets']):raise ValueError('Both source Coauthor datasets required')
    output=confined(root,args.output)
    if output.exists():raise ValueError('Fresh disabled Coauthor output required')
    output.mkdir();(output/'studies').mkdir();plan=make_plan(prepared,args.output,old_plan['host_bindings'])
    plan['immutable_source_bundle']={'package':args.runtime_prefix,'launcher_entrypoint':args.runtime_prefix+'/launcher.py',
        'model_worker_entrypoint':args.runtime_prefix+'/run_study.py','endpoint_worker_entrypoint':args.runtime_prefix+'/run_endpoint.py',
        'requires_root_runtime_resource_qualification_before_GO':True}
    for value,*_ in prepared:
        path=output/'studies'/(value['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(value)+'\n').encode(),mtime=0));path.chmod(0o444)
        plan['studies'].append({'path':str(path.relative_to(root)),'spec_hash':fingerprint(value)})
    keys=('initial_training_segments','all_run_records','full_budget_optimizer_updates','planned_TRAIN_route_forwards',
        'planned_validation_route_forwards','search_selection_jobs','bank_exports','scalar_CAL_fits','evaluation_jobs',
        'genuine_I4_banks','reused_I4_body0_artifacts','comparative_served_prediction_views')
    summary={'schema':'polyformer-coauthor-practical-disabled-preparation-v1','plan_hash':fingerprint(plan),'policy_hash':fingerprint(policy),
        'totals':{k:sum(c[k] for _,c,*_ in prepared) for k in keys},
        'per_study':[{'study_id':s['study_id'],'counts':c,'served_banks':b,'search_domains':d} for s,c,b,d in prepared],
        'source_dataset_metadata':source_metadata,'scientific_training_started':False,'scientific_launch_enabled':False,
        'GO_created':False,'full_v2_and_author_packets_preserved':True,'heldout_target_tensors_decoded':False,
        'prepared_label_files_verified_by_existing_custody_hash_only':True,'author_score_protocol_equality':False,
        'resource_status':'PROVISIONAL_REQUIRES_ROOT_QUALIFICATION'}
    for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]:
        path=output/name;path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');path.chmod(0o444)
    print(canonical({k:v for k,v in summary.items() if k!='per_study'}))


if __name__=='__main__':main()
