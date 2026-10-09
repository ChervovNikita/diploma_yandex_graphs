"""Disabled resident original-file complete-cohort readout. No host or model calls."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = 'd0b30e8297bbedc6063a0bf0c553869157165759d8cd2a41e93e4fe4324d0ea4'
PILOT = '4350d456c4f646738f0fc71268e7641c13dd4a97ef36c841262fbf2a872cd9d0'
ARMS = ['shared_own_only','native_pool_credit','source_view_supervision','uncoupled_source_contrast','COMMON_cycle','assigned_source_supply']
REFERENCES = ['plain_native','untied_same_six_factors']
PAIRS = [[1,1],[2,2],[3,3]]
NATIVE_SOURCE = 'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2'
ENTRY_SOURCES = {
    'shared18':'sehgnn_IMDB_paired_shared_source_entry_preparation_20261009_v1',
    'reference24':'sehgnn_IMDB_paired_independent_reference_entry_preparation_20261009_v1'}
OWNER_SOURCES = {
    'native_five':'sehgnn_IMDB_literal_five_seed_native_reference_activation_root_20261009_v2/OWNED_REFERENCE.py',
    'reference24':'sehgnn_IMDB_paired_independent_reference_activation_root_20261009_v1/OWNED_RUN.py',
    'shared18':'sehgnn_IMDB_paired_shared_source_activation_root_20261009_v1/OWNED_RUN.py'}


def phase_path(value,exists=True):
    path=Path(value)
    assert path.is_absolute() and path==path.resolve(strict=exists) and path.is_relative_to(HERE.parent)
    return path


def sha(path):
    value=hashlib.sha256()
    with phase_path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(phase_path(path).read_text())


def bound(row):
    assert set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int and row['bytes']>0
    path=phase_path(row['path'])
    assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    return path


def write(path,value):
    phase_path(path,exists=False).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load(name,path):
    assert name not in sys.modules
    path=phase_path(path)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def source_gate():
    seal=read(HERE/'SEAL.json')
    assert seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json')==seal['manifest_sha256']
    for row in read(HERE/'MANIFEST.json')['files']:
        path=phase_path(HERE/row['path']);assert path.is_relative_to(HERE) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    sources=read(HERE/'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path=phase_path(HERE.parent/row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    return sources


def custody(receipt):
    """Actual root observations link immutable launch/terminal and saved owners."""
    assert receipt['schema']=='root-IMDB-complete-cohort-actual-custody-v1'
    assert receipt['root_authorized'] is True and receipt['study_binding']==STUDY
    assert receipt['expected_cuda_uuid']=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    assert set(receipt['cohorts'])=={'native_five','reference24','shared18'}
    launches={}
    for cohort,row in receipt['cohorts'].items():
        launch_path=bound(row['launch']);terminal_path=bound(row['terminal'])
        assert launch_path.name=='LAUNCH.json' and terminal_path==launch_path.parent/'TERMINAL.json'
        launch=read(launch_path);terminal=read(terminal_path)
        assert launch['launcher_sha256']==sha(HERE.parent/OWNER_SOURCES[cohort])
        assert row['source_commit']==launch['source_commit'] and row['source_commit']
        assert terminal['exit_code']==0 and terminal['stop_reason'] is None
        assert terminal['child_reaped'] is True and terminal['original_pid_absent'] is True
        assert row['fresh_observation_UTC'] and row['actual_observation_evidence']
        bound(row['actual_observation_evidence'])
        for who in ('parent','child'):
            handle=row[who+'_handle']
            assert handle==launch[who] and set(handle)=={'pid','start_ticks','group','session','boot_id'}
            assert all(handle[key]>0 for key in ('pid','start_ticks','group','session')) and handle['boot_id']
            assert row[who+'_saved_handle_absent'] is True and row[who+'_saved_group_absent'] is True
        assert row['owned_cuda_residency_absent_on_expected_uuid'] is True
        assert row['timeout'] is False and row['retry'] is False
        assert launch['automatic_retry'] is False
        launches[cohort]=dict(record=launch,path=launch_path)
    return launches


def launch_owner(launch,release_binding,argv):
    """The observed owner launched this exact original release and output."""
    release_path=bound(release_binding)
    assert launch['path'].parent==release_path.parent, 'Custody launch is not the chosen original release owner'
    assert launch['record']['release_sha256']==release_binding['sha256'], 'Custody launch/release SHA mismatch'
    assert launch['record']['argv']==argv, 'Custody launch does not name the chosen source/release/output'


def entry_owner(cohort,entry_binding,entry,release_binding,release,family,launch,python):
    source=HERE.parent/ENTRY_SOURCES[cohort]
    assert bound(entry_binding)==phase_path(release['entry_output_directory'])/'ROOT_REPORT.json', 'Original entry report/output mismatch'
    assert phase_path(release['family_output_directory'])==family, 'Original release/selected family mismatch'
    assert release['enabled'] is True and release['source_seal_sha256']==PILOT
    action='fit_fixed_paired_shared_six_arm_source_family' if cohort=='shared18' else 'fit_fixed_paired_genuine_independent4_reference_family'
    assert entry['action']==release['action']==action
    assert release['root_entry_sha256']==entry['root_entry_sha256']==sha(source/'RUN.py')
    assert release['entry_source_seal_sha256']==entry['entry_source_seal_sha256']==sha(source/'SEAL.json')
    assert entry['release_sha256']==release_binding['sha256'] and entry['source_seal_sha256']==PILOT
    assert entry['study_binding']==release['study_binding']==STUDY
    assert hashlib.sha256(json.dumps(release['study'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==STUDY
    assert entry['family_output_directory']==str(family) and entry['pairs']==release['pairs']==PAIRS
    assert entry['status']=='complete' and entry['complete'] is True, 'Original '+cohort+' entry failed or incomplete'
    assert entry['startup_native_runtime_and_input_loading_inclusive'] is True
    for key in ('seconds','CPU_user_seconds','CPU_system_seconds','cumulative_RSS_peak_bytes'):
        assert type(entry[key]) in (int,float) and math.isfinite(entry[key]) and entry[key]>=0
    assert entry['seconds']<=release['resource_budget']['seconds']
    assert entry['cumulative_RSS_peak_bytes']<=release['resource_budget']['host_RSS_bytes']
    if cohort=='shared18':
        assert entry['arms']==release['arms']==ARMS
        assert entry['declared_shared_fits']==entry['completed_shared_fits']==release['actual_shared_fits']==18
        assert entry['native_and_independent_reference_competence_adopted'] is True
    else:
        assert entry['variants']==release['variants']==REFERENCES
        assert entry['declared_body_fits']==entry['completed_actual_body_fits']==release['actual_body_fits']==24
        assert entry['complete_groups']==6 and entry['reference_competence_pending'] is True
    launch_owner(launch,release_binding,[python,'-B',str(source/'RUN.py'),'--execute','--release',str(bound(release_binding))])


def family_owner(report,root,methods,method_key):
    assert report['status']=='complete' and report['complete'] is True and report['pairs']==PAIRS
    assert report[method_key+'s']==methods
    cells=report['cells']
    expected={(pair,method,str(root/('pair'+str(pair))/method)) for pair in (1,2,3) for method in methods}
    assert len(cells)==len(expected)
    assert {(row['role_seed'],row[method_key],row['output']) for row in cells}==expected
    assert all(row['base_seed']==row['role_seed'] and row['status']=='complete' and row['complete'] is True for row in cells)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--release',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'inactive':True,'outcomes_or_providers_read':False}));return
    assert args.release is not None
    started,usage=time.perf_counter(),resource.getrusage(resource.RUSAGE_SELF)
    sources=source_gate();cfg=read(args.release)
    assert cfg['enabled'] is True and cfg['root_source_review_approved'] is True and cfg['root_readout_approved'] is True
    assert cfg['source_seal_sha256']==sha(HERE/'SEAL.json') and cfg['entry_sha256']==sha(__file__)
    assert cfg['study_binding']==STUDY and cfg['pilot_source_seal_sha256']==PILOT
    assert cfg['schema']=='root-complete-closed-IMDB-quality-readout-release-v2'
    assert cfg['artifact_mode']=='resident_original_files' and cfg['transfer_scope']=='small_analysis_reports_only'
    output=phase_path(cfg['output_directory'],exists=False);assert not output.exists();output.mkdir(parents=True)
    report={'complete':False,'status':'started','quality_readout_complete':False,'scientific_success':False,'study_binding':STUDY}
    write(output/'ENTRY_REPORT.json',report)
    try:
        freeze=read(bound(cfg['quality_freeze']));assert sha(bound(cfg['quality_freeze']))==sources['quality_freeze_sha256']
        assert freeze['study_binding']==STUDY and freeze['pilot_source_seal_sha256']==PILOT and freeze['candidate']=='assigned_source_supply'
        shared=phase_path(cfg['shared_family_directory']);reference=phase_path(cfg['reference_family_directory']);native_root=phase_path(cfg['native_five_directory'])
        inventory=read(bound(cfg['closed_origin_inventory']));assert inventory['root_authorized'] is True and inventory['study_binding']==STUDY
        assert inventory['schema']=='root-closed-IMDB-selected-origin-inventory-v2'
        assert inventory['artifact_mode']=='resident_original_files'
        assert inventory['selected_checkpoint_digests_verified_at_original_locations_before_readout'] is True
        for row in inventory['files']:
            phase_path(row['path'])
        indexed={row['path']:row for row in inventory['files']}
        assert len(indexed)==len(inventory['files'])
        def file(root,name):
            row=indexed[str(root/name)];assert bound(row)==root/name;return root/name
        shared_release=read(bound(cfg['shared_fit_release']));reference_release=read(bound(cfg['reference_fit_release']))
        native_release=read(bound(cfg['native_fit_release']))
        shared_entry=read(bound(cfg['shared_entry_report']));reference_entry=read(bound(cfg['reference_entry_report']))
        owner_context=dict(owner_validation_status='pending',shared_entry=shared_entry,reference_entry=reference_entry,
            bindings={key:cfg[key] for key in ('shared_entry_report','reference_entry_report','shared_fit_release','reference_fit_release','native_fit_release')})
        write(output/'OWNER_ENTRY_CONTEXT.json',owner_context)
        status=[];records={};array_paths={}
        for root,methods in ((shared,ARMS),(reference,REFERENCES)):
            for pair in (1,2,3):
                for method in methods:
                    cell=root/('pair'+str(pair))/method
                    row=dict(pair=pair,method=method,status='started',complete=False);status.append(row)
                    try:
                        record=read(file(cell,'RESULT.json'));complete=read(file(cell,'COMPLETE.json'))
                        assert record['role_seed']==record['base_seed']==pair
                        assert record['arm' if root==shared else 'variant']==method
                        records[pair,method]=dict(result=record,result_binding=indexed[str(cell/'RESULT.json')],
                            diagnostic_binding=indexed[str(cell/'SELECTED_VALID_DIAGNOSTICS.pt')],cost_binding=indexed[str(cell/'COST_EVENTS.jsonl')],
                            selected_checkpoint_sha256=record['checkpoint_sha256'])
                        assert inventory['selected_origins'][str(cell)]==record['checkpoint_sha256']
                        file(cell,'COST_EVENTS.jsonl');array_paths[pair,method]=file(cell,'SELECTED_VALID_DIAGNOSTICS.pt')
                        row.update(status=record['status'],complete=record['complete'] is True and complete['complete'] is True and complete['status']=='complete')
                    except Exception as error:
                        row.update(status='missing_or_invalid',complete=False,failure=str(error))
        write(output/'COMPLETE_CELL_INVENTORY.json',status)
        assert len({(row['pair'],row['method']) for row in status})==24 and all(row['complete'] and row['status']=='complete' for row in status)
        shared_report=read(file(shared,'FAMILY_REPORT.json'));reference_report=read(file(reference,'FAMILY_REPORT.json'))
        native_cohort=read(file(native_root,'COHORT_REPORT.json'))
        original_costs=dict(shared_family=shared_report,reference_family=reference_report,
            shared_entry=shared_entry,reference_entry=reference_entry,
            native_five={key:native_cohort[key] for key in ('seconds','CPU_user_seconds','CPU_system_seconds','cumulative_RSS_peak_bytes')},
            nested_timings_not_summed=True)
        owner_context['original_costs']=original_costs
        write(output/'OWNER_ENTRY_CONTEXT.json',owner_context)
        launches=custody(read(bound(cfg['actual_custody_receipt'])))
        entry_owner('shared18',cfg['shared_entry_report'],shared_entry,cfg['shared_fit_release'],shared_release,shared,launches['shared18'],native_release['python_executable'])
        entry_owner('reference24',cfg['reference_entry_report'],reference_entry,cfg['reference_fit_release'],reference_release,reference,launches['reference24'],native_release['python_executable'])
        family_owner(shared_report,shared,ARMS,'arm');family_owner(reference_report,reference,REFERENCES,'variant')
        assert shared_report['complete'] is True and shared_report['complete_cells']==18 and shared_report['arms']==ARMS
        assert reference_report['complete'] is True and reference_report['complete_groups']==6 and reference_report['actual_body_fits']==24
        assert native_cohort['complete'] is True and native_cohort['status']=='complete' and native_cohort['mode']=='native_reference'
        assert native_cohort['native_five_seed_recipe_completed'] is True and native_cohort['seeds']==[1,2,3,4,5]
        assert native_cohort['release']==cfg['native_fit_release']
        assert native_release['enabled'] is True and native_release['action']=='fit_literal_five_seed_TRAIN_VALID_reference'
        assert native_release['seeds']==[1,2,3,4,5] and phase_path(native_release['output_directory'])==native_root
        assert native_cohort['source_seal_sha256']==native_release['source_seal_sha256']==sha(HERE.parent/NATIVE_SOURCE/'SEAL.json')
        launch_owner(launches['native_five'],cfg['native_fit_release'],[native_release['python_executable'],'-B',str(HERE.parent/NATIVE_SOURCE/'runner.py'),
            '--reference','--release',str(bound(cfg['native_fit_release'])),'--input-root',native_release['input_root'],'--output',str(native_root)])
        assert native_cohort['input_files']==native_release['expected_input_files']==shared_release['expected_input_files']==reference_release['expected_input_files']
        assert native_cohort['roles']==native_release['roles']
        assert shared_release['roles']==reference_release['roles']
        assert all(shared_release['roles'][seed]==native_release['roles'][seed] for seed in ('1','2','3'))
        for row in shared_release['roles'].values():
            bound(row)
        for key in ('native_qualification_receipt','integration_qualification_receipt','independent4_qualification_receipt'):
            assert shared_release[key]==reference_release[key]
            phase_path(shared_release[key]['path'])
        assert shared_release['reference_family_receipt']==cfg['reference_entry_report']
        assert shared_release['native_reference_receipt']==indexed[str(native_root/'COHORT_REPORT.json')]
        assert shared_entry['reference_family_receipt_sha256']==cfg['reference_entry_report']['sha256']
        assert shared_entry['native_reference_receipt_sha256']==indexed[str(native_root/'COHORT_REPORT.json')]['sha256']
        assert shared_entry['competence_adoption_receipt_sha256']==shared_release['competence_adoption_receipt']['sha256']
        native={}
        for seed in range(1,6):
            result=read(file(native_root/('seed'+str(seed)),'RESULT.json'))
            assert result['complete'] is True and result['status']=='complete'
            assert result['seed']==seed and result['native_full_VALID_BCE_selector'] is True
            assert result['identity']['input_files']==shared_release['expected_input_files']
            assert result['identity']['release_sha256']==cfg['native_fit_release']['sha256']
            if seed in (1,2,3):
                assert result['identity']['role_sha256s'][str(seed)]==shared_release['roles'][str(seed)]['sha256']
            valid=dict(result['fresh_selected']['fresh_scores']['VALID'])
            valid['micro_F1']=valid['served_micro_F1'];valid['macro_F1']=valid['served_macro_F1']
            native[seed]=dict(VALID=valid,selected_epoch=result['selected_epoch'],
                selected_checkpoint_sha256=result['selected_checkpoint_sha256'],record_binding=indexed[str(native_root/('seed'+str(seed))/'RESULT.json')],
                native_fresh_per_label_predictions='unavailable',costs={key:result[key] for key in ('seconds','CPU_user_seconds','CPU_system_seconds','cumulative_RSS_peak_bytes')})
            assert inventory['selected_origins'][str(native_root/('seed'+str(seed)))]==result['selected_checkpoint_sha256']
        adoption=read(bound(shared_release['competence_adoption_receipt']));assert adoption['native_and_independent_reference_competence_adopted'] is True
        assert adoption['root_authorized'] is True and adoption['study_binding']==STUDY
        assert adoption['reference_family_receipt']==cfg['reference_entry_report'] and adoption['native_reference_receipt']==indexed[str(native_root/'COHORT_REPORT.json')]
        assert adoption['roles']==shared_release['roles'] and adoption['expected_input_files']==shared_release['expected_input_files']
        checkpoint_origins={str(root/('pair'+str(pair))/method):name for root,methods,name in (
            (shared,ARMS,'SELECTED_BANK_STATE.pt'),(reference,REFERENCES,'SELECTED_INDEPENDENT4_STATE.pt')) for pair in (1,2,3) for method in methods}
        checkpoint_origins.update({str(native_root/('seed'+str(seed))):'SELECTED_STATE.pt' for seed in range(1,6)})
        assert set(inventory['selected_origins'])==set(inventory['selected_checkpoints'])==set(checkpoint_origins)
        for cell,name in checkpoint_origins.items():
            row=inventory['selected_checkpoints'][cell]
            assert row['path']==str(Path(cell)/name), 'Selected checkpoint is not at its original owning cell/seed path'
            assert bound(row)==Path(cell)/name and row['sha256']==inventory['selected_origins'][cell]
        write(output/'CHECKPOINT_ORIGIN_VERIFICATION.json',dict(artifact_mode='resident_original_files',
            selected_checkpoint_digests_verified_at_original_locations_before_readout=True,
            model_weights_loaded=False,selected_checkpoints=inventory['selected_checkpoints']))
        owner_context['owner_validation_status']='admitted_complete_exact_original_owners'
        write(output/'OWNER_ENTRY_CONTEXT.json',owner_context)
        # Only now import CPU array tools and the existing saved-analysis APIs.
        torch=importlib.import_module('torch');rt={'torch':torch,'device':torch.device('cpu')}
        phase=HERE.parent
        load('state_helpers',phase/'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2'/'state_helpers.py')
        engine=load('_readout_native_engine',phase/'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2'/'engine.py')
        metadata=load('_readout_original_metadata',phase/'sehgnn_grouped_bank_training_callback_source_20261009_v1'/'bank_training.py')
        sys.path.insert(0,str(phase/'sehgnn_paired_six_arm_source_supply_pilot_source_20261009_v2'))
        import family_driver
        import independent_controls
        import selected_diagnostics
        import analysis
        costs=engine.Costs(output,torch,rt['device'])
        pilot_cfg=family_driver.PilotConfig(enabled=True,root_source_review_approved=True,root_scientific_release_approved=True,
            source_seal_sha256=PILOT,native_qualification_binding=shared_release['native_qualification_receipt']['sha256'],
            integration_qualification_binding=shared_release['integration_qualification_receipt']['sha256'],
            independent4_qualification_binding=shared_release['independent4_qualification_receipt']['sha256'],native_and_independent_reference_competence_adopted=True,
            role_bindings={int(seed):row['sha256'] for seed,row in shared_release['roles'].items()},
            role_files={int(seed):row['path'] for seed,row in shared_release['roles'].items()},study_binding=STUDY)
        with costs.measure('reuse_complete_independent_control_comparisons'):
            independent_controls.compare_independent_controls(rt,engine,metadata,shared,reference,output/'INDEPENDENT_COMPARISONS',pilot_cfg)
        diagnostics={};comparisons={}
        with costs.measure('safe_saved_selected_diagnostic_array_reads'):
            for key,path in array_paths.items():
                item=torch.load(path,map_location='cpu',weights_only=True);assert item['paired_identity']['study_binding']==STUDY
                assert item['paired_identity']['role_seed']==key[0] and item['paired_identity']['base_seed']==key[0]
                assert item['paired_identity']['role_binding']==shared_release['roles'][str(key[0])]['sha256']
                diagnostics[key]=item
        for pair in (1,2,3):
            for method in ARMS[:-1]:
                comparisons[pair,method]=selected_diagnostics.compare_selected(torch,diagnostics[pair,method],diagnostics[pair,'assigned_source_supply'],costs,method)
            for method in REFERENCES:
                comparisons[pair,method]=torch.load(output/'INDEPENDENT_COMPARISONS'/('pair'+str(pair))/(method+'_POOL_CHANGES.pt'),map_location='cpu',weights_only=True)
        with costs.measure('thin_complete_tables_uncertainty_and_frozen_decisions'):
            result=analysis.summarize(torch,diagnostics,native,comparisons,records,original_costs,freeze,output)
        report.update(status='complete',complete=True,quality_readout_complete=True,
            advance_to_unused_confirmation=result['advance_to_independently_frozen_unused_confirmation'],scientific_success=False)
    except BaseException as error:
        report.update(status='failed_or_incomplete',complete=False,quality_readout_complete=False,failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        end=resource.getrusage(resource.RUSAGE_SELF)
        report.update(seconds=time.perf_counter()-started,CPU_user_seconds=end.ru_utime-usage.ru_utime,CPU_system_seconds=end.ru_stime-usage.ru_stime,
            cumulative_process_RSS_peak_bytes=int(end.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))
        write(output/'ENTRY_REPORT.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
