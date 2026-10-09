"""Disabled one-shot original-metadata assembly and unchanged V2 CPU readout."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=R/'experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BOOT='24c315a7-3c08-471f-b550-b9a3e1faf75d'
READER_NAME='sehgnn_IMDB_complete_cohort_quality_readout_preparation_20261009_v2'
STUDY='d0b30e8297bbedc6063a0bf0c553869157165759d8cd2a41e93e4fe4324d0ea4'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def guard():
    assert socket.gethostname()=='anogena-2-0', 'Wrong allocation hostname'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[GPU], 'Wrong allocation UUID inventory'


def phase_path(value,exists=True):
    path=Path(value)
    assert path.is_absolute() and path==path.resolve(strict=exists) and path.is_relative_to(PHASE)
    return path


def sha(path):
    digest=hashlib.sha256()
    with phase_path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path=phase_path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def bound(row):
    assert set(row)=={'path','bytes','sha256'} and binding(row['path'])==row
    return phase_path(row['path'])


def read(path):
    return json.loads(phase_path(path).read_text())


def write_new(path,value):
    with phase_path(path,exists=False).open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def source_gate():
    assert HERE.parent==PHASE
    seal=read(HERE/'SEAL.json')
    assert seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json')==seal['manifest_sha256']
    for row in read(HERE/'MANIFEST.json')['files']:
        path=phase_path(HERE/row['path']);assert path.is_relative_to(HERE) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    for row in read(HERE/'SOURCE_BINDINGS.json')['files']:
        path=phase_path(PHASE/row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    review=read(PHASE/'sehgnn_IMDB_complete_readout_root_repair_review_20261009_v1/REVIEW.json')
    assert review['review_status']=='accepted_for_publication_with_runtime_disabled'
    reader_path=PHASE/READER_NAME/'RUN.py'
    assert review['v2_entry_sha256']==sha(reader_path) and review['v2_seal_sha256']==sha(reader_path.parent/'SEAL.json')
    spec=importlib.util.spec_from_file_location('_original_closed_cohort_reader',reader_path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    reader.source_gate()
    return reader


def process_identity(pid,boot):
    try:
        text=Path('/proc',str(pid),'stat').read_text()
    except FileNotFoundError:
        return None
    fields=text[text.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]),boot_id=boot)


def observe_absence(launches):
    """One finite /proc snapshot and one CUDA query; never signal any workload."""
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==BOOT
    processes=[]
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            identity=process_identity(int(path.name),boot)
            if identity is not None:
                processes.append(identity)
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    cuda=[]
    for line in raw.splitlines():
        fields=[value.strip() for value in line.split(',')]
        assert len(fields)==3 and fields[1].isdigit(), 'Unparseable actual CUDA residency observation'
        cuda.append(dict(uuid=fields[0],pid=int(fields[1]),used_memory=fields[2]))
    evidence=dict(UTC=utc(),hostname=socket.gethostname(),expected_cuda_uuid=GPU,boot_id=boot,
        method='one finite /proc identity/group scan and one GPU compute-app query',cohorts={},
        unrelated_workloads_changed=False,scientific_results_or_arrays_opened=False)
    for cohort,launch in launches.items():
        observations={}
        for who in ('parent','child'):
            saved=launch[who];assert saved['boot_id']==boot
            present=process_identity(saved['pid'],boot)
            group=[row for row in processes if row['group']==saved['group']]
            assert present!=saved and not group, 'Original saved handle/group still present'
            observations[who]=dict(saved_handle=saved,current_pid_identity=present,saved_handle_absent=True,saved_group_members=group,saved_group_absent=True)
        pids={launch[who]['pid'] for who in ('parent','child')}
        owned=[row for row in cuda if row['uuid']==GPU and row['pid'] in pids]
        assert not owned, 'Saved owner PID still appears in expected CUDA residency'
        evidence['cohorts'][cohort]=dict(handles=observations,owned_cuda_rows=owned,owned_cuda_residency_absent_on_expected_uuid=True)
    return evidence


def assemble(reader,cfg,activation):
    releases={key:read(bound(row['release'])) for key,row in cfg['originals'].items()}
    launches={key:read(bound(row['launch'])) for key,row in cfg['originals'].items()}
    terminals={key:read(phase_path(row['launch']['path']).parent/'TERMINAL.json') for key,row in cfg['originals'].items()}
    for key,terminal in terminals.items():
        if cfg['originals'][key]['terminal'] is not None:
            bound(cfg['originals'][key]['terminal'])
        assert terminal['exit_code']==0 and terminal['stop_reason'] is None
        assert terminal['child_reaped'] is True and terminal['original_pid_absent'] is True
        assert launches[key]['automatic_retry'] is False and launches[key]['release_sha256']==cfg['originals'][key]['release']['sha256']
    native_release=releases['native_five'];shared_release=releases['shared18'];reference_release=releases['reference24']
    assert native_release['python_version']==cfg['qualified_python_version']==platform.python_version()
    assert native_release['python_executable']==cfg['qualified_python_executable']==str(Path(sys.executable).absolute())
    shared=phase_path(shared_release['family_output_directory']);reference=phase_path(reference_release['family_output_directory']);native=phase_path(native_release['output_directory'])
    entries={key:phase_path(releases[key]['entry_output_directory'])/'ROOT_REPORT.json' for key in ('shared18','reference24')}
    # These reports contain status/count/resource metadata, no outcome metrics.
    entry_reports={key:read(path) for key,path in entries.items()}
    for key,report in entry_reports.items():
        assert report['status']=='complete' and report['complete'] is True, 'Original entry failed or remains incomplete'
    assert entry_reports['shared18']['completed_shared_fits']==18
    assert entry_reports['reference24']['completed_actual_body_fits']==24 and entry_reports['reference24']['complete_groups']==6
    families={key:read(root/'FAMILY_REPORT.json') for key,root in (('shared18',shared),('reference24',reference))}
    reader.family_owner(families['shared18'],shared,reader.ARMS,'arm')
    reader.family_owner(families['reference24'],reference,reader.REFERENCES,'variant')
    assert families['shared18']['complete_cells']==18 and families['reference24']['complete_groups']==6 and families['reference24']['actual_body_fits']==24
    cells=[(root/('pair'+str(pair))/method,name) for root,methods,name in (
        (shared,reader.ARMS,'SELECTED_BANK_STATE.pt'),(reference,reader.REFERENCES,'SELECTED_INDEPENDENT4_STATE.pt')) for pair in (1,2,3) for method in methods]
    cells.extend((native/('seed'+str(seed)),'SELECTED_STATE.pt') for seed in range(1,6))
    assert native_release['seeds']==[1,2,3,4,5] and len(cells)==29
    for root in (native,phase_path(shared_release['entry_output_directory']),phase_path(reference_release['entry_output_directory'])):
        complete=read(root/'COMPLETE.json');assert complete['status']=='complete' and complete['complete'] is True
    for cell,_ in cells:
        complete=read(cell/'COMPLETE.json');assert complete['status']=='complete' and complete['complete'] is True
    # No scientific RESULT or checkpoint/array bytes have been read above.
    evidence=observe_absence(launches)
    write_new(activation/'ACTUAL_ABSENCE_OBSERVATION.json',evidence)
    custody=reader.read(PHASE/READER_NAME/'ROOT_CUSTODY_TEMPLATE_DISABLED.json')
    custody['root_authorized']=True
    for key,row in custody['cohorts'].items():
        launch=launches[key]
        row.update(launch=cfg['originals'][key]['launch'],terminal=binding(phase_path(cfg['originals'][key]['launch']['path']).parent/'TERMINAL.json'),
            source_commit=launch['source_commit'],fresh_observation_UTC=evidence['UTC'],actual_observation_evidence=binding(activation/'ACTUAL_ABSENCE_OBSERVATION.json'),
            owned_cuda_residency_absent_on_expected_uuid=True,timeout=False,retry=False)
        for who in ('parent','child'):
            row.update({who+'_handle':launch[who],who+'_saved_handle_absent':True,who+'_saved_group_absent':True})
    write_new(activation/'ACTUAL_CUSTODY.json',custody)
    owners=reader.custody(custody)
    for key,root in (('shared18',shared),('reference24',reference)):
        reader.entry_owner(key,binding(entries[key]),entry_reports[key],cfg['originals'][key]['release'],releases[key],root,owners[key],native_release['python_executable'])
    # Only after complete metadata and actual clean custody: hash original files.
    inventory=reader.read(PHASE/READER_NAME/'ORIGIN_INVENTORY_TEMPLATE_DISABLED.json')
    inventory.update(root_authorized=True,files=[],selected_origins={},selected_checkpoints={})
    files=[shared/'FAMILY_REPORT.json',reference/'FAMILY_REPORT.json',native/'COHORT_REPORT.json',native/'COMPLETE.json']+list(entries.values())
    for cell,name in cells:
        result=read(cell/'RESULT.json');assert result['status']=='complete' and result['complete'] is True
        expected=result['selected_checkpoint_sha256'] if name=='SELECTED_STATE.pt' else result['checkpoint_sha256']
        checkpoint=binding(cell/name);assert checkpoint['sha256']==expected
        inventory['selected_origins'][str(cell)]=expected;inventory['selected_checkpoints'][str(cell)]=checkpoint
        files.extend([cell/'RESULT.json',cell/'COMPLETE.json'])
        if name!='SELECTED_STATE.pt':
            files.extend([cell/'SELECTED_VALID_DIAGNOSTICS.pt',cell/'COST_EVENTS.jsonl'])
    assert len(inventory['selected_origins'])==len(inventory['selected_checkpoints'])==29
    inventory['files']=[binding(path) for path in files]
    inventory['selected_checkpoint_digests_verified_at_original_locations_before_readout']=True
    write_new(activation/'CLOSED_ORIGIN_INVENTORY.json',inventory)
    release=reader.read(PHASE/READER_NAME/'RELEASE.disabled.json')
    release.update(enabled=True,root_source_review_approved=True,root_readout_approved=True,source_seal_sha256=sha(PHASE/READER_NAME/'SEAL.json'),entry_sha256=sha(PHASE/READER_NAME/'RUN.py'),
        actual_custody_receipt=binding(activation/'ACTUAL_CUSTODY.json'),closed_origin_inventory=binding(activation/'CLOSED_ORIGIN_INVENTORY.json'),
        quality_freeze=binding(PHASE/'sehgnn_IMDB_paired_pilot_quality_root_freeze_20261009_v1/FREEZE.json'),
        shared_fit_release=cfg['originals']['shared18']['release'],reference_fit_release=cfg['originals']['reference24']['release'],native_fit_release=cfg['originals']['native_five']['release'],
        shared_entry_report=binding(entries['shared18']),reference_entry_report=binding(entries['reference24']),shared_family_directory=str(shared),reference_family_directory=str(reference),
        native_five_directory=str(native),output_directory=cfg['readout_output_directory'])
    write_new(activation/'READOUT_RELEASE.json',release)
    return native_release


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--activation-release',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True,allocation_queries_or_project_outcomes_read=False,reader_invoked=False)));return
    assert args.activation_release is not None
    guard()  # Always before project reads/writes or process observations.
    reader=source_gate();cfg=read(args.activation_release)
    assert cfg['schema']=='root-one-shot-IMDB-resident-readout-activation-v1' and cfg['enabled'] is True
    assert cfg['root_activation_source_review_approved'] is True and cfg['root_readout_execution_approved'] is True
    assert cfg['root_full18_native5_reference24_and_actual_custody_closure_confirmed'] is True
    assert cfg['source_seal_sha256']==sha(HERE/'SEAL.json') and cfg['entry_sha256']==sha(__file__)
    assert cfg['study_binding']==STUDY and set(cfg['originals'])=={'shared18','reference24','native_five'}
    assert cfg['originals']==read(HERE/'PRODUCER_BINDINGS.json')['originals']
    assert cfg['artifact_mode']=='resident_original_files' and cfg['transfer_scope']=='small_analysis_reports_only'
    assert cfg['automatic_retry'] is False
    activation=phase_path(cfg['activation_directory'],exists=False);output=phase_path(cfg['readout_output_directory'],exists=False)
    assert not activation.exists() and not output.exists() and activation!=output
    activation.mkdir(parents=True,exist_ok=False)  # Permanent one-shot marker; never reuse/retry.
    started,usage=time.perf_counter(),resource.getrusage(resource.RUSAGE_SELF)
    report=dict(status='started',complete=False,reader_invoked=False,automatic_retry=False,UTC=utc(),scientific_success=False)
    write_new(activation/'ACTIVATION_REPORT.json',report)
    try:
        native_release=assemble(reader,cfg,activation)
        env=dict(os.environ);env.update(native_release['runtime_environment']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        argv=[native_release['python_executable'],'-B',str(PHASE/READER_NAME/'RUN.py'),'--execute','--release',str(activation/'READOUT_RELEASE.json')]
        guard()
        write_new(activation/'CPU_READER_INVOCATION.json',dict(UTC=utc(),argv=argv,cwd=str(R),CUDA_VISIBLE_DEVICES='',reader_sha256=sha(PHASE/READER_NAME/'RUN.py'),attempts=1))
        with (activation/'reader.stdout').open('x') as stdout,(activation/'reader.stderr').open('x') as stderr:
            report['reader_invocation_attempted']=True
            result=subprocess.run(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,check=False)
        report['reader_invoked']=True
        report['reader_returncode']=result.returncode
        assert result.returncode==0, 'Existing reader failed; retain its failure and do not retry'
        completed=read(output/'ENTRY_REPORT.json')
        assert completed['complete'] is True and completed['quality_readout_complete'] is True
        report.update(status='complete',complete=True,quality_readout_complete=True)
    except BaseException as error:
        report.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        end=resource.getrusage(resource.RUSAGE_SELF)
        report.update(seconds=time.perf_counter()-started,CPU_user_seconds=end.ru_utime-usage.ru_utime,CPU_system_seconds=end.ru_stime-usage.ru_stime,
            cumulative_process_RSS_peak_bytes=int(end.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))
        (activation/'ACTIVATION_REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
