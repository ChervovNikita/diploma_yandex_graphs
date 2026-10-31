"""Pure-metadata successor of the complete disabled Coauthor packet.

Only a fixed execution option changes for two Physics procedures. All registered
learning, role, selection and independent-body records remain source declared.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path

from adapters.polyformer_recompute import METHOD_FORWARDS,execution_option
from compile_molecular_matrix import make_plan
from spec import canonical,fingerprint,validate


def receipt(root,path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def transform_study(original,*,old_prefix,new_prefix,runtime_files,lineage):
    value=copy.deepcopy(original);enabled=[];extra=0;weight=0.
    for run in value['runs']:
        if not run['adapter'].startswith(old_prefix+'/'):
            raise ValueError('Every prior Coauthor factory must use its exact frozen runtime')
        run['adapter']=new_prefix+run['adapter'][len(old_prefix):]
        if value['dataset']=='coauthor-physics' and run['kind']=='model_fit' and run['method'] in METHOD_FORWARDS:
            if run['config'].get('execution_engineering') is not None:raise ValueError('Prior packet has an execution option')
            if run['adapter']!=new_prefix+'/adapters/polyformer.py:make_adapter':raise ValueError('Exact source callback factory required')
            run['adapter']=new_prefix+'/adapters/polyformer_recompute.py:make_adapter'
            option=execution_option(run['method']);run['config']['execution_engineering']=option
            # Retain the source scheduling weight and add explicit replay work;
            # these are unmeasured relative weights, not seconds-per-fit claims.
            prior_weight=run['config']['estimated_seconds'];run['config']['estimated_seconds']=2*prior_weight
            weight+=prior_weight;replays=option['planned_full_replay_forwards_per_train_unit']*run['budget']['maximum']
            extra+=replays;enabled.append({'run_id':run['run_id'],'method':run['method'],'stage':run['config']['stage'],
              'split':run['split'],'seed':run['seed'],'budget_units':run['budget']['maximum'],
              'logical_forwards_per_unit':option['logical_route_forwards_per_train_unit'],
              'additional_complete_replay_forwards':replays,'prior_scheduling_weight':prior_weight,
              'new_scheduling_weight':run['config']['estimated_seconds']})
    outside=[r for r in value['source']['files'] if not r['path'].startswith(old_prefix+'/')]
    value['source']['files']=outside+runtime_files
    value['source']['route_recompute_successor']={'schema':'coauthor-source-execution-successor-lineage-v1',
      'previous_spec_hash':fingerprint(original),'previous_study_id':original['study_id'],'lineage':lineage,
      'fixed_execution_option_not_search_coordinate':True,'enabled_Physics_methods':list(METHOD_FORWARDS),
      'other_dataset_or_method_execution_changed':False,'source_objectives_and_hyperparameters_changed':False,
      'selector_parameter_IDs_and_rankings_changed':False,'original_GNNM_learning_records_changed':False,
      'replay_forward_work_additional':extra,'resource_and_concurrency_gate_still_required':True}
    expected=96 if value['dataset']=='coauthor-physics' else 0
    if len(enabled)!=expected:raise ValueError('Exact two48-fit Physics procedures required')
    # Normalize only declared runtime/execution/scheduling changes. Everything
    # else, including original GNNM records and all55 served methods, is exact.
    checked=copy.deepcopy(value)
    for run,prior in zip(checked['runs'],original['runs']):
        if run['run_id']!=prior['run_id']:raise ValueError('Original run order changed')
        run['adapter']=prior['adapter']
        if 'execution_engineering' in run['config']:
            del run['config']['execution_engineering'];run['config']['estimated_seconds']=prior['config']['estimated_seconds']
    checked['source']=copy.deepcopy(original['source'])
    if checked!=original:raise ValueError('Successor changed a learning/data/reference contract')
    validate(value)
    return value,{'enabled_fits':enabled,'additional_planned_replay_forwards':extra,
                  'additional_unmeasured_scheduling_weight':weight,'full_semantic_normalization_equals_prior':True}


def prepare(root,prior,packet,lineage):
    """Compile local or guarded remote metadata; no Torch/data-array import."""
    root=Path(root).resolve();prior=Path(prior).resolve();packet=Path(packet).resolve()
    if not prior.is_relative_to(root) or not packet.is_relative_to(root):raise ValueError('Owned project paths required')
    output=packet/'matrix'
    if output.exists():raise ValueError('Fresh immutable successor matrix required')
    prior_plan=json.loads((prior/'matrix/PLAN.json').read_text());prior_summary=json.loads((prior/'matrix/SUMMARY.json').read_text())
    if fingerprint(prior_plan)!=prior_summary['plan_hash'] or prior_plan['scientific_launch_enabled'] is not False:
        raise ValueError('Exact complete disabled prior packet required')
    new_prefix=str((packet/'runtime').relative_to(root));old_prefix=prior_plan['immutable_source_bundle']['package']
    runtime_files=[receipt(root,p) for p in sorted((packet/'runtime').rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md')]
    runtime_files+=[lineage]
    prepared=[];execution_rows=[];previous_proofs=[]
    for row in prior_plan['studies']:
        # The local mirrored lineage retains the exact original study bytes.
        path=prior/'matrix/studies'/Path(row['path']).name;original=json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(original)!=row['spec_hash']:raise ValueError('Original mirrored study changed')
        old_summary=next(r for r in prior_summary['per_study'] if r['study_id']==original['study_id'])
        value,work=transform_study(original,old_prefix=old_prefix,new_prefix=new_prefix,runtime_files=runtime_files,lineage=lineage)
        counts=copy.deepcopy(old_summary['counts']);counts['planned_TRAIN_replay_route_forwards']=work['additional_planned_replay_forwards']
        counts['planned_TRAIN_route_forwards_including_replay']=counts['planned_TRAIN_route_forwards']+counts['planned_TRAIN_replay_route_forwards']
        prepared.append((value,counts,copy.deepcopy(old_summary['served_banks']),copy.deepcopy(old_summary['search_domains'])))
        execution_rows.append({'study_id':value['study_id'],'dataset':value['dataset'],**work})
        previous_proofs.append({'study_id':original['study_id'],'spec_hash':row['spec_hash'],'exact_source_study_bytes':receipt(root,path),
          'records':len(original['runs']),'methods':len({r['method'] for r in original['runs']}),
          'original_GNNM_fits':sum(r['kind']=='model_fit' and r['config'].get('is_original_gnnm',False) for r in original['runs'])})
    if {s['dataset'] for s,*_ in prepared}!={'coauthor-cs','coauthor-physics'}:raise ValueError('Both new Coauthor datasets required')
    output.mkdir();(output/'studies').mkdir();relative=str(output.relative_to(root))
    plan=make_plan(prepared,relative,prior_plan['host_bindings'])
    plan['immutable_source_bundle']={'package':new_prefix,'launcher_entrypoint':new_prefix+'/launcher.py',
      'model_worker_entrypoint':new_prefix+'/run_study.py','endpoint_worker_entrypoint':new_prefix+'/run_endpoint.py',
      'requires_root_runtime_resource_qualification_before_GO':True}
    for study,*_ in prepared:
        path=output/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
        plan['studies'].append({'path':str(path.relative_to(root)),'spec_hash':fingerprint(study)})
    keys=list(prior_summary['totals'])+['planned_TRAIN_replay_route_forwards','planned_TRAIN_route_forwards_including_replay']
    summary=copy.deepcopy(prior_summary)
    summary.update(schema='polyformer-coauthor-recompute-disabled-successor-v1',plan_hash=fingerprint(plan),
      totals={k:sum(c[k] for _,c,*_ in prepared) for k in keys},
      per_study=[{'study_id':s['study_id'],'counts':c,'served_banks':b,'search_domains':d} for s,c,b,d in prepared],
      prior_packet_plan_hash=prior_summary['plan_hash'],execution_integration=execution_rows,
      complete_prior_record_preservation=previous_proofs,source_lineage=lineage,
      inherited_numerical_CPU_source_equivalence=True,numerical_CPU_equivalence_tests_repeated=False,
      source_and_data_arrays_decoded=False,CUDA_executions=0,scientific_fits=0,
      two_slot_concurrency_qualified=False,scientific_launch_enabled=False,GO_created=False,
      resource_status='PRIOR_RESERVATIONS_RETAINED_ROOT_GATE_REQUIRED;TWO_EXACT_RANDOM_CALLBACKS_MEASURED_ONLY')
    if summary['totals']['planned_TRAIN_replay_route_forwards']!=1152000:raise ValueError('Exact additional replay work required')
    for name,value in [('PLAN.json',plan),('SUMMARY.json',summary)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    entry={'id':'practical_polyformer_coauthor_recompute_v1','plan':receipt(root,output/'PLAN.json'),
           'summary':receipt(root,output/'SUMMARY.json'),'plan_hash':fingerprint(plan),
           'coverage':[lineage]}
    (packet/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n')
    return summary,entry


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','prior-local-mirror','packet','lineage'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.project_root).resolve()
    lineage=json.loads(Path(args.lineage).read_text())
    summary,entry=prepare(root,Path(args.prior_local_mirror),Path(args.packet),lineage)
    print(canonical({'entry':entry,'totals':summary['totals'],'CUDA':0,'fits':0,'GO':False}))
