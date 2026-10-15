"""Immutable metadata generator for the already proposed 2,890 BN migrations.

Not a launcher. It creates new composed JOBs and explicit downstream deltas;
it cannot certify runtime, reporting, resource, claim or human GO admission.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
E = HERE.parent
PHASE = E.parent
ROOT = PHASE.parents[1]
ALLOCATION = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
EXPECTED = {'fits':6460,'changed':2890,'unchanged':3570,'exports':270,'banks':5920,'controls':5110,
            'direct_G0':1690,'ogb_changed':2720,'link_changed':170,'interaction_changed':680}
BLOCKS = ['normalization_source_generation_root_review', 'normalized_producer_native_qualification',
          'composed_selected_serving_extra_state_successor_rebind', 'reporting_profiles_and_contrasts_rebind',
          'reporting_observer_genuine_composed_runtime_authority', 'dispatcher_release_and_ownership_rebind',
          'existing_claim_and_checkpoint_custody_review', 'measured_target_resource_cost_and_pair_admission',
          'exact_updated_campaign_human_GO']


def require(value, message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def receipt(path):
    path=Path(path).resolve();b=path.read_bytes()
    return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(pin, root):
    path=Path(pin['path']).resolve()
    require(path.is_relative_to(root) and path.suffix=='.json', 'Only exact project-confined JSON metadata may be opened')
    require(receipt(path)==pin,'Pinned metadata bytes changed: '+str(path))
    return json.loads(path.read_text())


def write(path, value):
    path=Path(path)
    with path.open('x') as stream: json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return receipt(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def guard():
    require(socket.gethostname()=='anogena-2-0','Wrong scientific host; no project reads')
    observed=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    require(observed==[UUID],'Wrong physical allocation; no project reads')
    require(ROOT==ALLOCATION and HERE==ALLOCATION/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/normalization_campaign_migration_v1',
            'Use the reviewed allocation deployment')


def sources(expected_sha):
    pin=receipt(HERE/'MANIFEST.json');require(pin['sha256']==expected_sha,'Exact root-reviewed source manifest required')
    manifest=read(pin,ROOT)
    require(manifest['schema']=='normalization-campaign-migration-source-v1','Wrong source schema')
    for row in manifest['files']:
        path=(HERE/row['path']).resolve();require(path.is_relative_to(HERE),'Source member escapes package')
        require(all(receipt(path)[k]==row[k]for k in('bytes','sha256')),'Source member changed')
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        path=(PHASE/row['path_relative_to_phase']).resolve();require(path.is_relative_to(ROOT),'Source dependency escapes repo')
        require(all(receipt(path)[k]==row[k]for k in('bytes','sha256')),'Frozen predecessor changed')
    return pin


def api():
    return module(E/'composable_normalization_v2/contracts.py','_normalization_campaign_contracts')


def affected(row, roster):
    cell=next((c for c in roster['cells']if c['id']==row['cell_id']),None)
    require(cell is not None,'Compiler preset absent from frozen roster')
    return row['track']=='long_convergence' and 'gnnm_private_bn_state_v3' in cell['shared_method_ids'] \
        and row['method_id'] in cell['shared_method_ids'] \
        and row['method_id'] not in ('gnnm_boundary_native','gnnm_private_bn_state_v3')


def composed_job(parent, parent_pin, roster, contracts, normalization_pin, banks_pin):
    declaration={'schema':contracts.SCHEMA,'policy':contracts.POLICY,
        'source_manifest_sha256':normalization_pin['sha256'],'banks_sha256':banks_pin['sha256'],
        'parent_descriptor':copy.deepcopy(parent_pin),'parent_job_sha256':contracts.digest(parent),
        'matched_control_method_id':contracts.CONTROL}
    return contracts.migrated_job(parent,declaration,roster)


def migrated_bank(bank, mapping, contracts):
    result=copy.deepcopy(bank)
    if not any(fid in mapping for fid in bank['fit_ids']): return result
    require(len(bank['fit_ids'])==1,'Native M1/I4 banks must not migrate')
    result['id']+=contracts.SUFFIX
    result['fit_ids']=[mapping[fid]for fid in bank['fit_ids']]
    result['normalization_policy']=contracts.POLICY
    result['fit_status']='composed_descriptor_prepared_not_admitted'
    return result


def dispatch_delta(task, candidate, job, child_pin, runtime_pin):
    require(task['kind']=='fit' and task['id']==candidate['id'] and candidate['admitted'] is False,
            'One unadmitted parent fit/candidate pair required')
    new_candidate=copy.deepcopy(candidate)
    new_candidate.update(id=job['id'],job_descriptor=child_pin,runtime_entrypoint=runtime_pin,
        runtime_job_hash=digest(job),canonical_runtime_job_hash=digest(job),admitted=False)
    new_task=copy.deepcopy(task)
    new_task.update(id=job['id'],source_descriptor=child_pin,runtime_job_hash=digest(job),
        canonical_runtime_job_hash=digest(job),science_contract_hash=job['expanded_contract_hash'],
        execution_candidate_id=job['id'],output_relative='fits/'+hashlib.sha256(job['id'].encode()).hexdigest(),
        runtime_binding=None,execution_binding=None,binding=None,resources=None,cost_case_prior=None,estimated_seconds=None,
        execution_status='blocked_normalization_successor_not_admitted')
    new_task['qualification_blocks']=list(dict.fromkeys([*task.get('qualification_blocks',[]),*BLOCKS]))
    # Old component evidence is retained as ancestry, never a composed-PASS claim.
    inherited=new_task.pop('functional_evidence_attachment',None)
    return {'parent_task':copy.deepcopy(task),'parent_candidate':copy.deepcopy(candidate),
        'successor_task':new_task,'successor_candidate':new_candidate,
        'parent_component_evidence_not_transferred_as_qualification':inherited,
        'pending_exact_execution_candidates_pin':True,'placement_and_group_preserved':True,
        'ownership_and_release_rebind_required':True}


def reporting_delta(spec, job, child_pin, runtime_pin, normalization_pin, producer_pin, bank_pin, control):
    # Preserve all common comparison authorities. The whole changed state owner
    # receives a new explicit readout/variant identity, without changing method aliases.
    require(spec['bindings']==spec['identity']['bindings'],'Parent reporting authorities disagree')
    identity=copy.deepcopy(spec['identity']);identity['fit_id']=job['id']
    identity['normalization_policy']=job['normalization_composition']['policy']
    identity['bindings']=copy.deepcopy(spec['identity']['bindings'])
    binding=digest({'parent_readout':spec['identity']['bindings']['readout'],
        'normalization_composition':job['normalization_composition'], 'producer_manifest':producer_pin,
        'genuine_runtime_entrypoint':runtime_pin,'job_hash':digest(job)})
    identity['bindings']['readout']=binding
    value={'parent_fit_id':spec['identity']['fit_id'],'fit_id':job['id'], 'descriptor':child_pin,
        'job_hash':digest(job),'runtime_entrypoint':runtime_pin,'identity':identity,
        'normalization_sources':{'composition_manifest':normalization_pin,'producer_manifest':producer_pin,'bank_law':bank_pin},
        'parent_spec_sha256':digest(spec),'preserved_common_bindings':{k:v for k,v in spec['bindings'].items()if k!='readout'},
        'parent_wrapper_authority':copy.deepcopy(spec['wrapper_authority']),
        'parent_observed_input_files':copy.deepcopy(spec['observed_input_files']),
        'matched_control_fit_id':control,'original_G0_retained_as_separate_total_comparison':True,
        'apply_to_frozen_profile_requires_new_reporting_source':True,'old_profile_or_contrast_plan_reused_unchanged':False}
    if 'same_bank_mean_identity' in spec:
        mean=copy.deepcopy(spec['same_bank_mean_identity']);mean.update(fit_id=job['id'],normalization_policy=identity['normalization_policy'])
        mean['bindings']=copy.deepcopy(identity['bindings']);mean['bindings']['readout']=digest({'actual_readout_hash':binding,'readout':'same_selected_bank_mean'})
        value['same_bank_mean_identity']=mean
    return value


def prepare(output, source_pin):
    inputs=json.loads((HERE/'INPUTS.json').read_text());require(inputs['schema']=='normalization-campaign-migration-inputs-v1','Wrong finite inputs')
    loaded={k:read(pin,ROOT)for k,pin in inputs['inputs'].items()}
    index=loaded['compiler_index'];control=loaded['controls'];banks=loaded['banks']
    release=loaded['portable_release'];candidates=loaded['portable_candidates'];reporting=loaded['reporting_index']
    require(len(index['fits'])==6460 and len(banks['logical_banks'])==5920 and len(control['entries'])==5110,'Parent compiler counts differ')
    require(len(release['tasks'])==6730 and len(candidates)==6730 and release['human_GO'] is False
            and release['all_tasks_qualified'] is False,'Only the exact unadmitted parent projection may migrate')
    rows={r['id']:r for r in index['fits']};tasks={t['id']:t for t in release['tasks']};execution={c['id']:c for c in candidates}
    require(len(rows)==6460 and set(tasks)==set(execution) and set(rows)=={t['id']for t in tasks.values()if t['kind']=='fit'},'Parent task identity coverage differs')
    roster=json.loads((E/'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text());contracts=api()
    counts=json.loads((E/'normalization_policy_decision_v1/COUNTS.json').read_text())
    eligible={fid:r for fid,r in rows.items()if affected(r,roster)}
    require(len(eligible)==counts['changed_scientific_fit_contracts']==2890,'Declared migration count differs')
    require(sum(r['family']=='ogb'for r in eligible.values())==2720 and sum(r['family']=='link'for r in eligible.values())==170,'Task-family migration changed')
    require(sum(r['method_id'].startswith('interaction_')for r in eligible.values())==680,'Interaction migration count differs')
    by_method=Counter(r['method_id']for r in eligible.values());by_preset=Counter(r['cell_id']for r in eligible.values())
    require(all(by_method[r['method_id']]==r['private_BN_candidate_fit_contracts_affected']for r in counts['per_recipe']),
            'Every proposed recipe application must migrate, including controls')
    require(all(by_preset[r['preset']]==r['candidate_fits_affected']for r in counts['per_preset']),
            'Per-preset normalization scope changed')
    mapping={fid:fid+contracts.SUFFIX for fid in eligible}
    matched={(r['cell_id'],r['split'],r['cohort_seed']):fid for fid,r in rows.items()if r['method_id']==contracts.CONTROL}
    require(len(matched)==170,'Retain the exact explicit private-BN-only controls')
    selected_control={fid:matched[(row['cell_id'],row['split'],row['cohort_seed'])]for fid,row in eligible.items()}
    profiles={}
    for pin in reporting['preset_files']:
        value=read(pin,ROOT);preset=value['registry']['presets'][0]['id'];require(preset not in profiles,'Duplicate parent profile')
        profiles[preset]=(pin,value)
    require(len(profiles)==27,'All 27 parent profiles required')
    specs={fid:spec for _,profile in profiles.values()for fid,spec in profile['fit_specs'].items()}
    require(set(specs)==set(rows) and sum(len(p['fit_specs'])for _,p in profiles.values())==6460,
            'Reporting fit inventory differs or duplicates a physical fit')
    require(sum(len(p['I4_specs'])for _,p in profiles.values())==270
            and sum('same_bank_mean_identity'in s for s in specs.values())==270,'I4 or same-bank diagnostic count differs')
    for _,profile in profiles.values():
        for spec in profile['I4_specs'].values():
            require(len(spec['component_fit_ids'])==4 and not any(fid in mapping for fid in spec['component_fit_ids']),
                    'I4 components must be unchanged native bodies')
    # Exact new owners; parent runtime/helper/checkpoint sources are not edited.
    normalization_pin=receipt(E/'composable_normalization_v2/MANIFEST.json')
    bank_pin=receipt(E/'private_bn_state_v3/banks.py')
    producer_pin=receipt(E/'composable_normalization_runtime_v1/MANIFEST.json')
    runtime_pin=receipt(E/'composable_normalization_runtime_v1/runtime/run_fit.py')
    sys.path.insert(0,str(E/'composable_normalization_runtime_v1'))
    descriptor=module(E/'composable_normalization_runtime_v1/descriptor_binding.py','_normalization_campaign_path_schema')
    children={};new_rows=[];dispatch=[];report_deltas=[];migration=[]
    output.mkdir();(output/'jobs').mkdir()
    try:
        for fid,row in rows.items():
            candidate=execution[fid];spec=specs[fid]
            require(candidate['job_descriptor']==spec['descriptor'] and candidate['runtime_job_hash']==row['runtime_job_hash']==spec['job_hash'],
                    'Compiler/projected/reporting job identity differs')
            if fid not in mapping:
                new_rows.append(copy.deepcopy(row));continue
            parent=read(candidate['job_descriptor'],ROOT)
            require(parent['id']==fid and digest(parent)==row['runtime_job_hash'] and parent['expanded_contract_hash']==row['expanded_contract_hash'],
                    'Exact canonical semantic parent required')
            child=composed_job(parent,candidate['job_descriptor'],roster,contracts,normalization_pin,bank_pin)
            require(child['id']==mapping[fid],'Child identity differs')
            descriptor.audit_runtime_paths(child,str(ROOT),str(PHASE.relative_to(ROOT)))
            child_pin=write(output/'jobs'/(hashlib.sha256(child['id'].encode()).hexdigest()+'.json'),child)
            children[fid]=(child,child_pin)
            successor=copy.deepcopy(row);successor.update(id=child['id'],descriptor=child_pin,runtime_job_hash=digest(child),
                expanded_contract_hash=child['expanded_contract_hash'],status='composed_descriptor_prepared_not_admitted',
                runtime_metadata_validation='exact_V2_contract_and_typed_path_source_check_only',launch_admitted=False)
            successor['normalization_parent']={'fit_id':fid,'canonical_descriptor':copy.deepcopy(row['descriptor']),
                'execution_descriptor':copy.deepcopy(candidate['job_descriptor']),'runtime_job_hash':row['runtime_job_hash'],
                'science_contract_hash':row['expanded_contract_hash'],'matched_control_fit_id':selected_control[fid]}
            new_rows.append(successor)
            dispatch.append(dispatch_delta(tasks[fid],candidate,child,child_pin,runtime_pin))
            report_deltas.append(reporting_delta(spec,child,child_pin,runtime_pin,normalization_pin,producer_pin,bank_pin,selected_control[fid]))
            migration.append({'parent_fit_id':fid,'fit_id':child['id'],'parent_identity':successor['normalization_parent'],
                'descriptor':child_pin,'job_hash':digest(child),'science_contract_hash':child['expanded_contract_hash']})
        new_banks=[migrated_bank(b,mapping,contracts)for b in banks['logical_banks']]
        new_controls=[contracts.migrated_controls(c,mapping,selected_control.get(c['fit_id']))for c in control['entries']]
        g0=sum(link.get('resolved_control')=='gnnm_boundary_native'for c in control['entries']if c['fit_id']in mapping for link in c['controls'])
        require(g0==1690,'Direct G0 attribution count differs')
        ids={r['id']for r in new_rows};require(len(ids)==6460 and len(children)==2890,'Replacement must preserve physical-fit count')
        require(len({b['id']for b in new_banks})==5920 and all(fid in ids for b in new_banks for fid in b['fit_ids']),'Logical bank identity coverage differs')
        require(all(fid in ids for c in new_controls for link in c['controls']for fid in link['fit_ids']),'Control refers to an absent fit')
        exports=[copy.deepcopy(t)for t in release['tasks']if t['kind']!='fit']
        require(len(exports)==270 and all(not any(fid in mapping for fid in t['dependencies'])for t in exports),'Export-only dependencies must remain unchanged')
        artifacts={}
        artifacts['index']=write(output/'DESCRIPTOR_INDEX.json',{'schema':index['schema'],'fits':new_rows})
        artifacts['banks']=write(output/'LOGICAL_BANKS.json',{'schema':banks['schema'],'logical_banks':new_banks})
        artifacts['controls']=write(output/'CONTROL_LINKS.json',{'schema':control['schema'],'entries':new_controls})
        artifacts['mapping']=write(output/'MIGRATION_MAP.json',{'schema':'normalization-fit-replacement-map-v1','entries':migration})
        artifacts['dispatcher_delta']=write(output/'DISPATCH_DELTA.json',{'schema':'normalization-unadmitted-dispatch-delta-v1',
            'parent_release':inputs['inputs']['portable_release'],'parent_candidates':inputs['inputs']['portable_candidates'],
            'changed_tasks':dispatch,'unchanged_task_ids':[fid for fid in tasks if fid not in mapping],
            'unchanged_I4_exports':exports,'parent_output_root_retained_for_custody_review':release['output_root'],
            'new_admitted_release_created':False,'ownership_rebind_required':True})
        artifacts['reporting_delta']=write(output/'REPORTING_DELTA.json',{'schema':'normalization-reporting-rebind-delta-v1',
            'parent_index':inputs['inputs']['reporting_index'],'parent_profiles':[pin for pin,_ in profiles.values()],
            'fit_updates':report_deltas,'unchanged_fit_ids':[fid for fid in rows if fid not in mapping],
            'parent_counts_retained':copy.deepcopy(reporting['counts']), 'old_contrast_plans_cannot_be_adopted_unchanged':True,
            'future_control_links':artifacts['controls'],'reporting_registry_generated':False,
            'selected_serving_source_rebind_pending':{
                'required_manifest':receipt(E/'composable_normalization_final_scoring_v2/MANIFEST.json'),
                'root_binding_complete':False,'legacy_inherited_composed_scorer_must_not_be_bound':True}})
        result={'schema':'normalization-campaign-migration-preparation-v1','status':'metadata_generated_not_admitted',
            'source_manifest':source_pin,'parent_inputs':inputs['inputs'],'artifacts':artifacts,
            'counts':EXPECTED,'unchanged_fit_scientific_definitions_preserved':True,'I4_body0_M1_and_source_references_preserved':True,
            'changed_descriptor_native_settings_losses_and_callbacks_preserved':True,
            'remaining_dependencies':BLOCKS,'new_scientific_fits':0,'TEST_payload_accessed':False,
            'models_constructed':0,'Torch_imported':'torch'in sys.modules,'source_predecessors_modified':False,
            'full_campaign_GO':False,'runnable_release_or_reporting_registry_created':False,'server77_enabled':False}
        require('torch'not in sys.modules,'Metadata generation must not import Torch')
        write(output/'PREPARATION.json',result)
        return result
    except BaseException as error:
        write(output/'FAILED.json',{'status':'failed','error':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(limit=12),'completed_new_descriptors':len(children),'complete_generation':False,
            'native_work':0,'full_campaign_GO':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();guard()
    output=Path(args.output).resolve()
    require(output.parent==HERE and not output.exists(),'Use a fresh direct child generation inside this source package')
    pin=sources(args.source_manifest_sha256)
    started=time.perf_counter();result=prepare(output,pin)
    print(json.dumps({'preparation':receipt(output/'PREPARATION.json'),'counts':result['counts'],
        'elapsed_seconds':time.perf_counter()-started,'native_work':0,'full_campaign_GO':False}))


if __name__=='__main__':main()
