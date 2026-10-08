"""All12 bounded CPU bundles, no development scoring or selection."""
import argparse
import os
import signal
import subprocess
import sys
import time
from common import *
from heads import prepare, paired_start, fit


def proc(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), group=int(fields[2]), session=int(fields[3]))
    except FileNotFoundError:
        return None


def stop(child, owner, hard_deadline):
    if child.poll() is not None:
        return True
    if proc(child.pid) != owner:
        return False
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=max(0,min(5,hard_deadline-time.monotonic())))
    except subprocess.TimeoutExpired:
        if proc(child.pid) != owner:
            return False
        os.killpg(child.pid, signal.SIGKILL)
        try: child.wait(timeout=max(0,hard_deadline-time.monotonic()))
        except subprocess.TimeoutExpired: return False
    return child.poll() is not None


def worker(path, digest):
    started = time.monotonic()
    cpu_started = time.process_time()
    require(sha(path) == digest, 'Exact parent worker job')
    job = read(path)
    seal(ROOT, job['source_manifest_sha256'])
    cfg = read(bound(job['root_release']))
    require(cfg.get('enabled') is True and cfg.get('stage') == 'fit' and cfg.get('root_execution_authorized') is True,
            'Worker requires admitted root family')
    require(job['owner'] == proc(os.getppid()) and job['row'] in roster(), 'Owned parent and exact12 roster')
    os.environ.update(cpu_env())
    import numpy as np
    import torch
    signature = backend(torch, np)
    qualification = read(bound(cfg['backend_qualification']))
    require(qualification.get('passed') is True and qualification['backend'] == signature
            and qualification['source_manifest_sha256'] == job['source_manifest_sha256'], 'Exact short CPU backend qualification')
    output = (PHASE / job['output_directory']).resolve()
    require(output.is_relative_to(PHASE) and not output.exists(), 'Fresh owned endpoint output')
    output.mkdir(mode=0o700)
    try:
        a = load_arrays(np, job['prepared'])
        endpoint, receipt = fit(torch, np, a, job['row']['condition'], job['active_deadline_monotonic'])
        path = output / 'endpoint.npz'; np.savez(path, **endpoint)
        receipt.update(row=job['row'], endpoint=binding(path), prepared=job['prepared'], backend=signature,
                       inclusive_wall_seconds=time.monotonic()-started, CPU_process_seconds=time.process_time()-cpu_started,
                       development_scores_opened=False)
        write(output / 'RESULT.json', receipt)
    except Exception as error:
        write(output / 'FAILURE.json', dict(row=job['row'], error_type=type(error).__name__, error=str(error),
            inclusive_wall_seconds=time.monotonic()-started, automatic_retry=False, endpoint_substituted=False))
        raise


def family(path, digest):
    started = time.monotonic()
    cfg, output = release(path, digest, 'fit')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU family must hide CUDA')
    output.mkdir(mode=0o700)
    rows = [{**row, 'status':'not_attempted', 'unavailable_reason':'Not reached'} for row in roster()]
    assets = {}
    fatal = None
    family_deadline = started + 46790
    try:
        import numpy as np
        import torch
        signature = backend(torch, np)
        qualification = read(bound(cfg['backend_qualification']))
        require(qualification.get('passed') is True and qualification['backend'] == signature
                and qualification['source_manifest_sha256'] == cfg['source_manifest_sha256'], 'Exact CPU backend fixture')
        feature_seal = read(bound(cfg['feature_seal']))
        require(feature_seal.get('closed') is True and feature_seal['source_manifest_sha256'] == cfg['source_manifest_sha256'], 'Closed exact native feature collection')
        collection = read(bound(feature_seal['collection']))
        expected = [(r['arm'],r['seed']) for r in read(ROOT/'SOURCE_BINDINGS.json')['selected_custody']]
        require([(r['arm'],r['seed']) for r in collection['rows']] == expected, 'All nine fixed banks including failures')
        require(collection.get('TEST_access') is False and collection.get('refits') == 0, 'Untouched source features')
        for bank in collection['rows']:
            key = (bank['arm'], bank['seed'])
            try:
                require(time.monotonic() < family_deadline and bank['status'] == 'complete', 'Complete native bank within family bound')
                if bank['arm'] == 'be_unit_contrastive': bound(bank['native_cohorts'])
                a = prepare(np, load_arrays(np, bank['features']))
                qualification_pair = paired_start(torch, a) if bank['arm'] == 'be_unit_contrastive' else None
                archive = output / (bank['cell'] + '_prepared.npz'); np.savez(archive, **a)
                assets[key] = dict(prepared=binding(archive), source=bank, paired_start=qualification_pair)
            except Exception as error:
                assets[key] = dict(error_type=type(error).__name__, error=str(error))
        for row in rows:
            asset = assets[(row['arm'],row['seed'])]
            row['asset'] = asset
            if 'prepared' not in asset:
                row.update(status='source_unavailable', unavailable_reason=asset)
                continue
            if family_deadline-time.monotonic() < 3600:
                fatal = 'Finite whole-family bound cannot admit another complete bundle'; break
            began = time.monotonic()
            job = dict(row={key:row[key] for key in ('condition','seed','bundle','arm','members')},
                prepared=asset['prepared'], owner=proc(os.getpid()), root_release=binding(path),
                source_manifest_sha256=cfg['source_manifest_sha256'],
                output_directory=str((output/row['bundle']).relative_to(PHASE)),
                active_deadline_monotonic=began+3590, active_seconds=3590, cleanup_seconds=10, hard_seconds=3600)
            job_path = output / (row['bundle']+'_JOB.json'); write(job_path, job)
            with (output/(row['bundle']+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable,'-B',str(ROOT/'run_refits.py'),'--worker-job',str(job_path),
                    '--worker-sha256',sha(job_path)], env=cpu_env(), start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                owner = proc(child.pid)
                require(owner is not None and owner['group'] == owner['session'] == child.pid, 'Actual fresh worker session')
                row['owner'] = owner
                try:
                    code = child.wait(timeout=max(0,3590-(time.monotonic()-began)))
                    row['exit_code'] = code
                except subprocess.TimeoutExpired:
                    row.update(status='timeout', unavailable_reason='Original active cap; no restart or shortened fit')
                    if not stop(child,owner,began+3600):
                        fatal = 'Owned cleanup custody unavailable'; break
                    row['exit_code'] = child.poll()
            row['reaped'] = child.poll() is not None
            row['inclusive_wall_seconds'] = time.monotonic()-began
            result = output/row['bundle']/'RESULT.json'
            if row['status'] != 'timeout' and row['exit_code'] == 0 and result.is_file():
                receipt = read(result)
                require(receipt['row'] == job['row'] and receipt['prepared'] == asset['prepared'], 'Exact endpoint identity')
                bound(receipt['endpoint'])
                row.update(status=receipt['status'], result=binding(result), endpoint=receipt['endpoint'], unavailable_reason=None)
            elif row['status'] != 'timeout':
                row.update(status='failed', unavailable_reason='Owned worker failed; retained log/FAILURE')
    except Exception as error:
        fatal = type(error).__name__ + ': ' + str(error)
    finally:
        for row in rows:
            if row['status'] == 'not_attempted': row['unavailable_reason'] = fatal or row['unavailable_reason']
            for label, candidate in (('worker_job',output/(row['bundle']+'_JOB.json')),
                                     ('worker_log',output/(row['bundle']+'.log')),
                                     ('failure',output/row['bundle']/'FAILURE.json')):
                if candidate.is_file(): row[label] = binding(candidate)
        terminal = all(row.get('reaped') is True for row in rows if 'owner' in row)
        closure = dict(schema='Wiki24-direct12-fit-closure-v1', closed=True, whole12_accounted=True, rows=rows,
            roster=roster(), source_manifest_sha256=cfg['source_manifest_sha256'], feature_seal=cfg['feature_seal'],
            fatal=fatal, inclusive_wall_seconds=time.monotonic()-started, family_active_seconds=46790, family_hard_seconds=46800,
            all_owned_workers_reaped=terminal, development_scores_opened=False, TEST_access=False, automatic_retry=False)
        write(output/'CLOSURE.json', closure)
        write(output/'SEAL.json', dict(closed=True, closure=binding(output/'CLOSURE.json'), whole12_accounted=True,
                                      source_manifest_sha256=cfg['source_manifest_sha256'], scores_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path); parser.add_argument('--release-sha256')
    parser.add_argument('--worker-job', type=Path); parser.add_argument('--worker-sha256')
    args = parser.parse_args()
    if args.worker_job: worker(args.worker_job,args.worker_sha256)
    else: family(args.release,args.release_sha256)
