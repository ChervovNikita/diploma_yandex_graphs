"""One finite physical-closure wait, exact bind, full9/18 fit and45CAL readout."""
import argparse
import os
from pathlib import Path
import time
from fullrun_owner_v1 import R, P, H, S, UUID, LIMITS, QUEUE_SECONDS, READOUT_SECONDS, SEEDS, ARMS, SOURCE_MANIFEST_SHA256, EXECUTION_FILES, route, item, read, write, stamp, identity, cuda_processes, own
from bind_fullfit_v1 import bind

PENN = P / 'native_SAGE_Penn94_data_preparation_root_20261011_v1'
PHYSICS = P / 'native_SAGE_Physics_confirmation_root_20261011_v1'


def validate_root(path):
    path = Path(path).resolve()
    assert path == H / 'ROOT_PIPELINE_ADMISSION_V1.json'
    root = read(path)
    expected = dict(schema_version=1, status='ROOT_FIXED_FiLM_PIPELINE_ADMITTED', study_kind='native_PyG_FiLM_WikiCS',
        study_id='native_PyG_FiLM_WikiCS_20261011_v1', hostname='anogena-2-0', gpu_uuid=UUID,
        banks=9, optimizer_bundles=18, native_body_records=18, calibration_attempts=45,
        seeds=SEEDS, arms=ARMS, fit_caps=LIMITS, readout_wall_seconds=READOUT_SECONDS,
        queue_wait_seconds=QUEUE_SECONDS, TEST_access=False, retry=False, seed_subset=False,
        closure_rule='physical_terminal_closure_of_at_least_one_current_acquisition_cohort_any_outcome',
        cohort_owner_ends=[str(PENN / 'native_acquisition_v1_owner/OWNER_END.json'), str(PHYSICS / 'external_cohort_owner_v1/OWNER_END.json')])
    assert all(root.get(key) == value for key, value in expected.items())
    assert root['source_manifest'] == item(S / 'MANIFEST.json') and root['source_manifest']['sha256'] == SOURCE_MANIFEST_SHA256
    assert set(root['execution_sources']) == set(EXECUTION_FILES)
    for name, row in root['execution_sources'].items(): assert item(H / name) == row
    return root


def closed_cohort():
    cuda = cuda_processes()
    for cohort, directory, side in [('Penn94', PENN, 'native_acquisition_v1_owner'), ('Physics', PHYSICS, 'external_cohort_owner_v1')]:
        endpath = directory / side / 'OWNER_END.json'; startpath = endpath.with_name('OWNER_START.json')
        if not endpath.exists() or not startpath.exists(): continue
        try: end, start = read(endpath), read(startpath)
        except ValueError: continue  # A writer may still be completing its terminal JSON.
        if end.get('direct_child_wait') is not True or type(end.get('exit_code')) is not int: continue
        receipts = [item(startpath), item(endpath)]
        if cohort == 'Penn94':
            if end.get('stage') != 'acquire' or end.get('child_pid_absent') is not True or end.get('owned_cuda_pid_absent') is not True: continue
            if start.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip(): continue
            if start.get('child') != end.get('child'): continue
            pids = [end['child']['PID']]
            if identity(start['owner']['PID']) is not None: continue
        else:
            if end.get('child_pid_absent') is not True or start.get('child_pid') != end.get('child_pid'): continue
            inner = directory / 'cohort_owner_v1/OWNER_START.json'
            if not inner.exists(): continue
            children = read(inner).get('children', {})
            if set(children) != {'core', 'author'}: continue
            pids = [end['child_pid']] + [children[key]['pid'] for key in ('core', 'author')]
            receipts.append(item(inner))
        if not all(type(pid) is int and pid > 0 and identity(pid) is None and pid not in cuda for pid in pids): continue
        return dict(UTC=stamp(), cohort=cohort, physically_closed=True, acquisition_PIDs=pids, receipts=receipts,
            terminal_exit_code=end['exit_code'], outcome_gate=False, scientific_success_consulted=False,
            physical_proc_and_CUDA_absence=True, score_files_opened=False,
            current_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    return None


def pipeline(path):
    route(); root = validate_root(path)
    for name in ('SCIENTIFIC_CONFIG_V1.json', 'SCIENTIFIC_ADMISSION_V1.json', 'PHYSICAL_ACQUISITION_CLOSURE_V1.json',
                 'fit_owner_v1', 'readout_owner_v1', 'actual_family_v1', 'complete_analysis_v1', 'PIPELINE_START_V1.json', 'PIPELINE_END_V1.json'):
        assert not (H / name).exists(), 'Prior attempt preserved; this pipeline has no retry: ' + name
    began = time.monotonic(); phase = 'physical_acquisition_queue'; failure = None; success = False
    write(H / 'PIPELINE_START_V1.json', dict(UTC=stamp(), queue_owner=identity(os.getpid()),
        admission=item(path), source_manifest=root['source_manifest'], execution_sources=root['execution_sources'],
        queue_wait_seconds=QUEUE_SECONDS, fit_caps=LIMITS, readout_wall_seconds=READOUT_SECONDS,
        banks=9, optimizer_bundles=18, native_body_records=18, calibration_attempts=45,
        prior_failures_preserved=True, TEST_access=False, retry=False, seed_subset=False))
    try:
        while True:
            route()
            gate = closed_cohort()
            if gate is not None: break
            remaining = QUEUE_SECONDS - (time.monotonic() - began)
            if remaining <= 0: raise TimeoutError('24h physical acquisition closure wait expired; no fit or readout attempted')
            time.sleep(min(30, remaining))
        write(H / 'PHYSICAL_ACQUISITION_CLOSURE_V1.json', gate)
        phase = 'exact_fullfit_binding'; bind(path)
        phase = 'full9_bank18_fit_acquisition'
        if own('fit') != 0: raise RuntimeError('Full acquisition failure retained; no partial readout or retry')
        endpath = H / 'fit_owner_v1/OWNER_END.json'; end = read(endpath)
        assert end['status'] == 'COMPLETE_EXIT_0' and end['exit_code'] == 0 and end['direct_child_wait']
        assert end['child_pid_absent'] and end['owned_cuda_pid_absent'] and identity(end['child']['PID']) is None
        assert end['child']['PID'] not in cuda_processes()
        complete = H / 'actual_family_v1/COMPLETE_FAMILY.json'
        assert item(complete)['sha256'] == end['complete_family_sha256']
        spec = read(S / 'READER_INPUT_TEMPLATE.json')
        spec.update(study_id=root['study_id'], family_root=str(H / 'actual_family_v1'),
            reader_source=item(S / 'analysis.py'), protocol=item(S / 'PROTOCOL.json'),
            config=item(H / 'SCIENTIFIC_CONFIG_V1.json'), complete_family=item(complete), owner_end=item(endpath), math_reader=item(S / 'math_utils.py'))
        write(H / 'COMPLETE_READER_INPUT_SPEC_V1.json', spec)
        phase = 'one45_CAL_complete_readout'
        if own('readout') != 0: raise RuntimeError('Complete readout failure retained; no retry or final refit')
        success = True; phase = 'complete_full_pipeline'
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    fit_end = read(H / 'fit_owner_v1/OWNER_END.json') if (H / 'fit_owner_v1/OWNER_END.json').exists() else {}
    readout_end = read(H / 'readout_owner_v1/OWNER_END.json') if (H / 'readout_owner_v1/OWNER_END.json').exists() else {}
    result = dict(UTC=stamp(), status='COMPLETE_EXIT_0' if success else 'FAILED_RETAINED', phase=phase,
        seconds=time.monotonic()-began, failure=failure, admission=item(path), complete_pipeline=success,
        fit_owner_end=item(H / 'fit_owner_v1/OWNER_END.json') if (H / 'fit_owner_v1/OWNER_END.json').exists() else None,
        readout_owner_end=item(H / 'readout_owner_v1/OWNER_END.json') if (H / 'readout_owner_v1/OWNER_END.json').exists() else None,
        physical_gate=item(H / 'PHYSICAL_ACQUISITION_CLOSURE_V1.json') if (H / 'PHYSICAL_ACQUISITION_CLOSURE_V1.json').exists() else None,
        expected_banks=9, expected_optimizer_bundles=18, expected_native_body_records=18, expected_calibration_attempts=45,
        banks=fit_end.get('banks'), optimizer_bundles=fit_end.get('optimizer_bundles'),
        native_body_records=fit_end.get('native_body_records'), calibration_attempts=readout_end.get('calibration_attempts'),
        TEST_access=False, retry=False, seed_subset=False, favorable_result_required=False)
    write(H / 'PIPELINE_END_V1.json', result)
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--admission', type=Path, required=True)
    raise SystemExit(pipeline(parser.parse_args().admission))
