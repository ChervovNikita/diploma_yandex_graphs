"""Complete native Collab split conversion, reusing six valid outputs.

No model or prediction work. Original preparation failure and inputs remain.
The existing provider's worker and final contract builder are reused unchanged.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import resource
import signal
import subprocess
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PARENT = EXPANSION / 'final_role_population_preparation_v1'
PARENT_SHA = 'cfb4f6a3e672ecb6325cfbd738c9520b0a72e3a732274b7e3b5791a62d362f9b'
ARCHIVE_SHA = 'c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc'


def main():
    spec = importlib.util.spec_from_file_location('_original_population_preparation', PARENT / 'prepare.py')
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)  # stdlib definitions only, no data/native imports
    route = p.guard()
    source, pins = p.source_authority(PARENT_SHA)
    p.require(HERE == p.EXPANSION / 'population_completion_v2', 'Use the allocation copy')
    own_manifest = p.read_json(HERE / 'MANIFEST.json')
    for row in own_manifest['files']:
        f = HERE / row['path']
        p.require(f.stat().st_size == row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest() == row['sha256'],
                  'Completion source changed')
    own_pin = p.receipt(HERE / 'MANIFEST.json')
    roster, bindings, policy = p.load_inputs()
    from native_link import bind
    p.link_population = bind(p)
    if sys.argv[1:] == ['--worker']:
        p.worker('ogbl-collab', PARENT / 'allocation_population_v3/ogbl-collab', PARENT_SHA)
        return
    p.require(not sys.argv[1:], 'Unexpected arguments')
    old = PARENT / 'allocation_population_v1'
    prior_failure = p.read_json(old / 'FAILURE.json')
    p.require(prior_failure['status'] == 'failed' and prior_failure['TEST_scored'] is False,
              'Preserved initial failure required')
    out = PARENT / 'allocation_population_v3'
    p.require(not out.exists(), 'Preserve prior outputs; no automatic retry')
    out.mkdir()
    started, active = time.monotonic(), None
    try:
        admission = p.admission(out)
        os.sched_setaffinity(0, set(admission['selected_CPU_affinity']))
        resource.setrlimit(resource.RLIMIT_AS, (p.LIMITS['supervisor_address_space_bytes'],
                                               p.LIMITS['worker_address_space_bytes']))
        def deadline(signum, frame):
            raise TimeoutError('Population completion wall bound reached')
        def terminated(signum, frame):
            raise InterruptedError('Population completion interrupted')
        signal.signal(signal.SIGALRM, deadline)
        signal.signal(signal.SIGTERM, terminated)
        signal.setitimer(signal.ITIMER_REAL, 1200)
        p.write_json(out / 'START.json', {'route': route, 'parent_source': source, 'completion_source': own_pin,
                     'prior_failure': p.receipt(old / 'FAILURE.json'), 'admission': admission,
                     'started_at_UTC': p.utc(), 'TEST_metadata_access_planned': True, 'TEST_scored': False})
        reuse = []
        for dataset in p.DATASETS:
            if dataset == 'ogbl-collab':
                continue
            original = old / dataset / 'RESULT.json'
            value = p.read_json(original)
            p.require(value['status'] == 'complete' and value['dataset'] == dataset
                      and value['source_manifest'] == source and value['TEST_scored'] is False,
                      'Only exact completed original provider results may be reused')
            (out / dataset).mkdir()
            # Copy compact metadata only; ordered arrays remain at their existing
            # immutable receipt paths. finish() revalidates all source/ID bytes.
            (out / dataset / 'RESULT.json').write_bytes(original.read_bytes())
            reuse.append({'dataset': dataset, 'original': p.receipt(original),
                          'copied_metadata': p.receipt(out / dataset / 'RESULT.json'),
                          'provider_rerun': False, 'original_wall_seconds': value['wall_seconds']})
        reuse_pin = p.write_json(out / 'REUSED_SIX.json', reuse)
        previous = PARENT / 'allocation_population_v2'
        p.require(p.read_json(previous / 'FAILURE.json')['status'] == 'failed', 'Preserved conversion failure required')
        extraction_record = p.read_json(previous / 'ORIGINAL_TEST_EXTRACTION.json')
        p.require(extraction_record['archive']['sha256'] == ARCHIVE_SHA, 'Wrong retained official archive')
        for pin in [extraction_record['archive'], extraction_record['destination'],
                    *extraction_record['matching_native_roles'].values()]:
            p.verify(pin)
        extraction = p.write_json(out / 'EXISTING_ORIGINAL_TEST.json', {
            'original_extraction': p.receipt(previous / 'ORIGINAL_TEST_EXTRACTION.json'),
            'verified_native_files': extraction_record, 'new_extraction_or_download': False,
            'conversion_failure': p.receipt(previous / 'FAILURE.json'), 'TEST_scored': False})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false')
        env.pop('PYTHONPATH', None)
        command = [sys.executable, '-B', str(HERE / 'complete.py'), '--worker']
        launched = time.monotonic()
        with (out / 'ogbl-collab_WORKER.log').open('xb') as log, (out / 'ogbl-collab_RESOURCE_SAMPLES.jsonl').open('x') as samples:
            active = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                state = p.resource_state(active.pid)
                samples.write(json.dumps({'elapsed_seconds': time.monotonic() - launched, **state}) + '\n')
                samples.flush()
                p.require(state.get('RSS_bytes', 0) <= p.LIMITS['worker_RSS_stop_bytes'], 'Worker RSS bound exceeded')
                p.require(state['available_bytes'] >= p.LIMITS['minimum_live_available_bytes'], 'Insufficient live RAM headroom')
                p.require(time.monotonic() - launched < p.LIMITS['worker_wall_seconds'], 'Worker wall bound exceeded')
                time.sleep(1)
            code = active.wait()
            active = None
        p.write_json(out / 'ogbl-collab_EXIT.json', {'returncode': code, 'wall_seconds': time.monotonic() - launched})
        p.require(code == 0, 'Collab worker failed; preserve log and partial population')
        index = p.finish(out, roster, bindings, policy, source,
                         pins + [own_pin, p.receipt(__file__), p.receipt(HERE / 'native_link.py'), reuse_pin, extraction])
        index_pin = p.write_json(out / 'ROLE_CONTRACT_INDEX.json', index)
        p.write_json(out / 'RESULT.json', {'status': 'complete', 'index': index_pin, 'counts': index['counts'],
            'source_manifest': source, 'completion_source': own_pin, 'six_completed_populations_reused': True,
            'original_failure': p.receipt(old / 'FAILURE.json'), 'prior_failed_supervisor_wall_seconds': prior_failure['wall_seconds'] + p.read_json(previous / 'FAILURE.json')['wall_seconds'],
            'wall_seconds': time.monotonic() - started, 'TEST_accessed': True, 'TEST_scored': False,
            'models_constructed': 0, 'predictions_created': 0, 'training_updates': 0})
        files = [p.receipt(f) for f in sorted(out.rglob('*')) if f.is_file()]
        manifest = p.write_json(out / 'OUTPUT_MANIFEST.json', {'schema': 'native-final-population-output-manifest-v1',
                      'source_manifest': source, 'completion_source': own_pin, 'files': files})
        print(json.dumps({'status': 'complete', 'output_manifest': manifest, 'index': index_pin}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stopped = None
        if active is not None and active.poll() is None:
            stopped = active.pid
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=5)
        import traceback
        p.write_json(out / 'FAILURE.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc(),
                     'wall_seconds': time.monotonic() - started, 'TEST_metadata_access': 'possibly partial; see retained ledgers',
                     'TEST_scored': False, 'terminated_own_worker': stopped, 'no_retry': True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
