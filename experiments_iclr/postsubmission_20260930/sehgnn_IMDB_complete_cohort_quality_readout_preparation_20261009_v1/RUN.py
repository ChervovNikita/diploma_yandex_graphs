"""Disabled local, complete-cohort saved-output readout. No server or model calls."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = 'd0b30e8297bbedc6063a0bf0c553869157165759d8cd2a41e93e4fe4324d0ea4'
PILOT = '4350d456c4f646738f0fc71268e7641c13dd4a97ef36c841262fbf2a872cd9d0'
ARMS = ['shared_own_only','native_pool_credit','source_view_supervision','uncoupled_source_contrast','COMMON_cycle','assigned_source_supply']
REFERENCES = ['plain_native','untied_same_six_factors']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(row):
    path = Path(row['path'])
    assert path.is_absolute() and not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return path


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load(name,path):
    assert name not in sys.modules
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def source_gate():
    seal=read(HERE/'SEAL.json')
    assert seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json')==seal['manifest_sha256']
    for row in read(HERE/'MANIFEST.json')['files']:
        path=HERE/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    sources=read(HERE/'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path=HERE.parent/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    return sources


def custody(receipt):
    """Actual root observations link immutable launch/terminal and saved owners."""
    assert receipt['schema']=='root-IMDB-complete-cohort-actual-custody-v1'
    assert receipt['root_authorized'] is True and receipt['study_binding']==STUDY
    assert receipt['expected_cuda_uuid']=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    assert set(receipt['cohorts'])=={'native_five','reference24','shared18'}
    for row in receipt['cohorts'].values():
        launch=read(bound(row['launch']));terminal=read(bound(row['terminal']))
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--release',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'inactive':True,'outcomes_or_providers_read':False}));return
    started,usage=time.perf_counter(),resource.getrusage(resource.RUSAGE_SELF)
    sources=source_gate();cfg=read(args.release)
    assert cfg['enabled'] is True and cfg['root_source_review_approved'] is True and cfg['root_readout_approved'] is True
    assert cfg['source_seal_sha256']==sha(HERE/'SEAL.json') and cfg['entry_sha256']==sha(__file__)
    assert cfg['study_binding']==STUDY and cfg['pilot_source_seal_sha256']==PILOT
    output=Path(cfg['output_directory']);assert output.is_absolute() and not output.exists();output.mkdir(parents=True)
    report={'complete':False,'status':'started','quality_readout_complete':False,'scientific_success':False,'study_binding':STUDY}
    write(output/'ENTRY_REPORT.json',report)
    try:
        freeze=read(bound(cfg['quality_freeze']));assert sha(bound(cfg['quality_freeze']))==sources['quality_freeze_sha256']
        assert freeze['study_binding']==STUDY and freeze['pilot_source_seal_sha256']==PILOT and freeze['candidate']=='assigned_source_supply'
        custody(read(bound(cfg['actual_custody_receipt'])))
        shared=Path(cfg['shared_family_directory']);reference=Path(cfg['reference_family_directory']);native_root=Path(cfg['native_five_directory'])
        inventory=read(bound(cfg['closed_origin_inventory']));assert inventory['root_authorized'] is True and inventory['study_binding']==STUDY
        # This inventory contains only explicit local result/diagnostic/cost
        # files, selected-state SHA origins and all declared complete cells.
        assert inventory['selected_checkpoint_digests_verified_before_transfer'] is True
        indexed={row['path']:row for row in inventory['files']}
        def file(root,name):
            row=indexed[str(root/name)];assert bound(row)==root/name;return root/name
        status=[];records={};array_paths={}
        for root,methods in ((shared,ARMS),(reference,REFERENCES)):
            for pair in (1,2,3):
                for method in methods:
                    cell=root/('pair'+str(pair))/method
                    row=dict(pair=pair,method=method,status='started',complete=False);status.append(row)
                    try:
                        record=read(file(cell,'RESULT.json'));complete=read(file(cell,'COMPLETE.json'))
                        records[pair,method]=dict(result=record,result_binding=indexed[str(cell/'RESULT.json')],
                            diagnostic_binding=indexed[str(cell/'SELECTED_VALID_DIAGNOSTICS.pt')],cost_binding=indexed[str(cell/'COST_EVENTS.jsonl')],
                            selected_checkpoint_sha256=record['checkpoint_sha256'])
                        assert inventory['selected_origins'][str(cell)]==record['checkpoint_sha256']
                        file(cell,'COST_EVENTS.jsonl');array_paths[pair,method]=file(cell,'SELECTED_VALID_DIAGNOSTICS.pt')
                        row.update(status=record['status'],complete=record['complete'] and complete['complete'])
                    except Exception as error:
                        row.update(status='missing_or_invalid',complete=False,failure=str(error))
        write(output/'COMPLETE_CELL_INVENTORY.json',status)
        assert len({(row['pair'],row['method']) for row in status})==24 and all(row['complete'] and row['status']=='complete' for row in status)
        shared_report=read(file(shared,'FAMILY_REPORT.json'));reference_report=read(file(reference,'FAMILY_REPORT.json'))
        assert shared_report['complete'] is True and shared_report['complete_cells']==18 and shared_report['arms']==ARMS
        assert reference_report['complete'] is True and reference_report['complete_groups']==6 and reference_report['actual_body_fits']==24
        native_cohort=read(file(native_root,'COHORT_REPORT.json'));assert native_cohort['complete'] is True and native_cohort['status']=='complete'
        assert native_cohort['native_five_seed_recipe_completed'] is True and native_cohort['seeds']==[1,2,3,4,5]
        shared_release=read(bound(cfg['shared_fit_release']));assert shared_release['study_binding']==STUDY and shared_release['source_seal_sha256']==PILOT
        assert native_cohort['source_seal_sha256']=='5ca6dff2c1f64bf37e2642fff7101e4377a48e087d2b21082a8d46887dc3654e'
        assert native_cohort['input_files']==shared_release['expected_input_files']
        native={}
        for seed in range(1,6):
            result=read(file(native_root/('seed'+str(seed)),'RESULT.json'))
            assert result['complete'] is True and result['status']=='complete'
            assert result['seed']==seed and result['native_full_VALID_BCE_selector'] is True
            assert result['identity']['input_files']==shared_release['expected_input_files']
            if seed in (1,2,3):
                assert result['identity']['role_sha256s'][str(seed)]==shared_release['roles'][str(seed)]['sha256']
            native[seed]=dict(VALID=result['fresh_selected']['fresh_scores']['VALID'],selected_epoch=result['selected_epoch'],
                selected_checkpoint_sha256=result['selected_checkpoint_sha256'],record_binding=indexed[str(native_root/('seed'+str(seed))/'RESULT.json')],
                native_fresh_per_label_predictions='unavailable',costs={key:result[key] for key in ('seconds','CPU_user_seconds','CPU_system_seconds','cumulative_RSS_peak_bytes')})
            assert inventory['selected_origins'][str(native_root/('seed'+str(seed)))]==result['selected_checkpoint_sha256']
        adoption=read(bound(shared_release['competence_adoption_receipt']));assert adoption['native_and_independent_reference_competence_adopted'] is True
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
        original_costs=dict(shared_family=shared_report,reference_family=reference_report,
            shared_entry=read(bound(cfg['shared_entry_report'])),reference_entry=read(bound(cfg['reference_entry_report'])),
            native_five={key:native_cohort[key] for key in ('seconds','CPU_user_seconds','CPU_system_seconds','cumulative_RSS_peak_bytes')},
            nested_timings_not_summed=True)
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
