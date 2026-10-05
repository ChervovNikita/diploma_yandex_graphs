#!/usr/bin/env python3
"""Own three serialized fresh native300 comparator fits; never read outcomes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from common import (GPU_UUID, HERE, PHASE, REPO, file_row, load_source,
                    output_bytes, require, sha, utc, write)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == REPO and socket.gethostname() == 'anogena-2-0',
            'Authorized repository/hostname differs')
    require(HERE.resolve().is_relative_to(PHASE) and args.release.resolve().is_relative_to(PHASE)
            and sha(args.release) == args.release_sha256, 'Reviewed source/release route differs')
    release = json.loads(args.release.read_text())
    require(release.get('root_source_review_approved') is True
            and release.get('scientific_comparator_authorized') is True
            and release.get('TEST_authorized') is False, 'Explicit scientific release absent')
    require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256'], 'Approved source manifest differs')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        require(sha(HERE / row['path']) == row['sha256'], 'Reviewed packet bytes changed')
    plan = json.loads((HERE / 'PLAN.json').read_text())
    require(sha(HERE / 'PLAN.json') == release['plan_sha256']
            and release['caps'] == plan['caps_proposed_for_root_review']
            and plan['seeds'] == [0, 1, 2], 'Frozen three-seed cohort/limits differ')
    require(args.output.resolve() == (PHASE / release['execution_output_phase_relative']).resolve()
            and args.output.resolve().is_relative_to(PHASE) and not args.output.exists(),
            'Require the uniquely released fresh cohort output inside phase')
    owned_source = PHASE / 'pencil_citeseer_zero_update_resource_execution_20261005_v1/supervisor.py'
    require(sha(owned_source) == '18653b51d647a027e2195bd18fbfbc8eb72ee07341298cead79bb94910224924',
            'Previously reviewed owned-process supervision source changed')
    owned = load_source('pencil_native300_owned_process_primitives', owned_source)
    freeze_path = PHASE / 'citeseer_endpoint_frame_paired_development_20261005_v1/COHORT_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    require(freeze.get('complete') is True and len(freeze['completed_physical_fits']) == 36
            and freeze['plan_sha256'] == 'fd9fec451a81d0512cd8431ba2a990c580592ff6a6e8e12d784386a99c5e9daa',
            'Exact completed36-fit metadata gate required; no outcomes opened')
    args.output.mkdir()
    cohort_start = time.monotonic()
    receipt = dict(schema='pencil_native300_owned_serial_cohort_v1', status='RUNNING', UTC=utc(),
                   host=socket.gethostname(), GPU_UUID=GPU_UUID, release_sha256=args.release_sha256,
                   source_manifest_sha256=sha(HERE / 'MANIFEST.json'), seeds=[0, 1, 2],
                   completed_fits=[], failures=[], signals_sent=[], TEST_access=False,
                   comparative_outcomes_opened=False, prior_NCN_outcomes_opened=False,
                   native_recipe_shortened=False, automatic_retry=False, other_jobs_signalled=False,
                   co_resident_timing=True, cohort_freeze_sha256=sha(freeze_path))
    child = None
    identities = {}
    try:
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(PHASE / 'pencil_one_gpu_dependency_overlay_20261005_v1')
                + ':' + str(PHASE / 'native_ncn_dependency_overlay_20261005_v1')
                + ':' + str(REPO / '.venv/lib/python3.11/site-packages'),
                RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', CUDA_VISIBLE_DEVICES='0',
                PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                WANDB_MODE='disabled', OUTDATED_IGNORE='1')
        for seed in [0, 1, 2]:
            require(time.monotonic() - cohort_start < release['caps']['cohort_wall_seconds'],
                    'Cohort wall ceiling reached before next seed; do not dispatch')
            parent = args.output / ('seed_' + str(seed))
            parent.mkdir()
            output = parent / 'run01'
            command = [str(PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'),
                    str(HERE / 'worker.py'), '--release', str(args.release),
                    '--release-sha256', args.release_sha256, '--seed', str(seed), '--output', str(output)]
            # Numeric imports, data reads and constructors remain in the fresh child.
            memory_info = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, value = line.split(':', 1)
                memory_info[key] = int(value.strip().split()[0]) * 1024
            query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=30)
            devices = [line.split(',') for line in query.stdout.strip().splitlines()]
            require(len(devices) == 1 and devices[0][0].strip() == GPU_UUID,
                    'Immediate authorized singleton GPU UUID differs')
            gpu_free = int(devices[0][1].strip()) * 1024 ** 2
            gate = dict(UTC=utc(), seed=seed, GPU_UUID=GPU_UUID, fresh_GPU_free_bytes=gpu_free,
                    minimum_GPU_free_bytes=plan['dispatch_eligibility']['minimum_free_GPU_bytes'],
                    fresh_CPU_available_bytes=memory_info['MemAvailable'],
                    minimum_CPU_available_bytes=plan['dispatch_eligibility']['minimum_CPU_available_bytes'],
                    utilization_percent=int(devices[0][3].strip()), full_fit_readiness_unmeasured=True)
            write(parent / 'DISPATCH_GATE.json', gate)
            require(gpu_free >= gate['minimum_GPU_free_bytes']
                    and memory_info['MemAvailable'] >= gate['minimum_CPU_available_bytes'],
                    'Fresh dispatch headroom insufficient; preserve gate and stop without retry')
            log = (parent / 'worker_stdout_stderr.log').open('xb')
            child_start = time.monotonic()
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            identity = owned.proc(child.pid)
            require(identity is not None and identity['sid'] == identity['pgid'] == child.pid,
                    'Directly owned fresh worker session identity absent')
            identities = {str(child.pid): identity}
            fit = dict(seed=seed, status='RUNNING', UTC=utc(), command=command,
                       child_identity=identity, signals_sent=[], max_owned_RSS_bytes=0,
                       max_output_including_log_bytes=0, observed_owned_identities=identities)
            while True:
                live = owned.owned_session(child.pid, identities)
                fit['max_owned_RSS_bytes'] = max(fit['max_owned_RSS_bytes'], sum(row['RSS_bytes'] for row in live))
                fit['max_output_including_log_bytes'] = max(fit['max_output_including_log_bytes'], output_bytes(parent))
                fit.update(elapsed_child_seconds=time.monotonic() - child_start,
                           live_owned_process_count=len(live), observed_owned_identities=identities)
                failure = None
                caps = release['caps']
                if fit['max_owned_RSS_bytes'] > caps['max_parent_plus_live_descendant_RSS_bytes']:
                    failure = 'OWNED_RSS_CAP'
                if fit['max_output_including_log_bytes'] > caps['max_fit_retained_and_temporary_output_bytes']:
                    failure = 'OWNED_OUTPUT_RETENTION_CAP'
                if fit['elapsed_child_seconds'] > caps['per_fit_wall_seconds']:
                    failure = 'PER_FIT_WALL_CAP'
                if time.monotonic() - cohort_start > caps['cohort_wall_seconds']:
                    failure = 'COHORT_WALL_CAP'
                progress_path = output / 'PROGRESS.json'
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    fit['latest_progress_metadata'] = progress
                    for key in ('max_cuda_allocated_bytes', 'max_cuda_reserved_bytes'):
                        if progress.get(key, 0) > caps[key]:
                            failure = key
                if failure:
                    fit['signals_sent'] = owned.stop_owned(child, identities, failure)
                    fit.update(status='FAIL_RESOURCE_PRESERVED', reason=failure)
                    break
                exit_code = child.poll()
                if exit_code is not None:
                    fit['child_exit_code'] = exit_code
                    if owned.owned_session(child.pid, identities):
                        fit['signals_sent'] = owned.stop_owned(child, identities, 'WORKER_ENDED_WITH_DESCENDANTS')
                    custody_path = output / 'FINAL_CUSTODY.json'
                    if exit_code == 0 and custody_path.exists():
                        custody = json.loads(custody_path.read_text())
                        require(custody['complete'] is True and custody['seed'] == seed,
                                'Worker success without exact final custody')
                        # Read bytes/hashes only: keep comparisons closed until full cohort.
                        for row in custody['files']:
                            artifact = output / row['path']
                            require(artifact.stat().st_size == row['bytes'] and sha(artifact) == row['sha256'],
                                    'Retained scientific artifact custody differs')
                        fit.update(status='COMPLETE_FROZEN_NATIVE300_FIT', final_custody_sha256=sha(custody_path),
                                   retained_artifact_metadata=custody['files'])
                    else:
                        fit['status'] = 'FAIL_CHILD_PRESERVED'
                    break
                write(parent / 'SUPERVISOR_FIT_RECEIPT.json', fit)
                receipt.update(active_seed=seed, elapsed_cohort_seconds=time.monotonic() - cohort_start)
                write(args.output / 'COHORT_PROGRESS.json', receipt)
                time.sleep(1)
            fit.update(terminal_UTC=utc(), child_exit_code=child.wait(timeout=1))
            write(parent / 'SUPERVISOR_FIT_RECEIPT.json', fit)
            receipt['signals_sent'].extend(fit['signals_sent'])
            if fit['status'] != 'COMPLETE_FROZEN_NATIVE300_FIT':
                receipt['failures'].append(dict(seed=seed, reason=fit['status']))
                raise RuntimeError('Frozen comparator member failed; cohort stops without retry or score reading')
            receipt['completed_fits'].append(dict(seed=seed, final_custody_sha256=fit['final_custody_sha256'],
                    fit_receipt_sha256=sha(parent / 'SUPERVISOR_FIT_RECEIPT.json'),
                    elapsed_child_seconds=fit['elapsed_child_seconds']))
            child = None
            identities = {}
        require([row['seed'] for row in receipt['completed_fits']] == [0, 1, 2], 'Whole cohort incomplete')
        receipt.update(status='COMPLETE_THREE_SEED_NATIVE300_COHORT', complete=True,
                       terminal_UTC=utc(), inclusive_cohort_seconds=time.monotonic() - cohort_start)
        write(args.output / 'COHORT_FREEZE.json', receipt)
    except BaseException as error:
        receipt.update(status='FAIL_COHORT_PRESERVED_NO_RETRY', complete=False,
                       error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        if child is not None:
            receipt['signals_sent'].extend(owned.stop_owned(child, identities, 'COHORT_SUPERVISOR_EXCEPTION'))
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        write(args.output / 'COHORT_FAILURE.json', receipt)
        raise
    finally:
        receipt.update(terminal_UTC=utc(), inclusive_cohort_seconds=time.monotonic() - cohort_start)
        write(args.output / 'TERMINAL_RECEIPT.json', receipt)


if __name__ == '__main__':
    main()
