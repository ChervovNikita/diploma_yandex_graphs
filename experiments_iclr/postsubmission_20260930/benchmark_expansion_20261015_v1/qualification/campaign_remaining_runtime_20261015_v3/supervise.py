"""One root-admitted allocation check, held shared slot, no wait/retry or Git action."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from witness_support import E, ROOT, HERE, GPU, GiB, guard, inside, limits, module, own_source, read, receipt, require, verify, write


def own_processes(child_pid, process):
    values = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                if os.getpgid(int(path.name)) == child_pid:
                    value = process(int(path.name))
                    if value:
                        values.append(value)
            except ProcessLookupError:
                pass
    return values


def gpu_budget(state, peers, own, cap):
    require(state['uuid'] == GPU and len(state['processes']) <= (2 if own else 1), 'Wrong GPU or third model worker')
    by_pid = {p['pid']: p for p in peers}
    own_ids = {(p['pid'], p['start_ticks']) for p in own}
    growth, own_bytes = 0, 0
    for row in state['processes']:
        if (row['pid'], row['start_ticks']) in own_ids:
            own_bytes += row['gpu_memory_bytes']
        else:
            peer = by_pid.get(row['pid'])
            require(peer is not None and row['start_ticks'] == str(peer['start_ticks'])
                    and row['gpu_memory_bytes'] <= peer['peak_GPU_bytes'], 'Unadmitted GPU peer or growth beyond envelope')
            growth += peer['peak_GPU_bytes'] - row['gpu_memory_bytes']
    require(own_bytes <= cap, 'Own process group exceeded GPU reservation')
    if not own:
        require(state['free_bytes'] - growth >= cap, 'Insufficient GPU headroom after peer growth')
    return growth, own_bytes


def execute(admission_path):
    guard()  # Before metadata reads, output writes or native imports.
    admission = read(inside(admission_path))
    require(admission.get('schema') == 'remaining-runtime-root-admission-v1'
            and admission.get('status') == 'ADMITTED' and admission.get('root_reviewed_source') is True
            and admission.get('full_campaign_GO') is False
            and admission.get('TEST_calls') == admission.get('automatic_retries') == 0,
            'Explicit root-reviewed finite admission required')
    case = admission['case']
    admitted_updates = 38 if case == 'bound_observer_molbace' else 0
    require(admission.get('TRAIN_updates') == admitted_updates, 'Exact case update budget required')
    cap = limits(case)
    require(admission['resources'] == cap, 'Exact finite resource limits required')
    inputs = own_source(admission['source_manifest_sha256'])
    require(admission['INPUTS_sha256'] == receipt(HERE / 'INPUTS.json')['sha256'], 'Exact frozen inputs required')
    output = inside(admission['output'])
    require(output.is_relative_to(E/'qualification') and not output.exists(), 'One fresh qualification output required')
    output.mkdir(parents=True)
    started = time.monotonic()
    child = lock = launched = None
    peak_rss = peak_gpu = 0
    report = {'schema': 'remaining-runtime-supervisor-v1', 'case': case, 'status': 'preflight',
        'admission': receipt(admission_path), 'native_child_started': False, 'automatic_retries': 0,
        'TRAIN_updates_admitted': admitted_updates, 'TEST_accessed': False, 'full_campaign_GO': False, 'root_supervisor_pid': os.getpid()}
    def save():
        report['wall_seconds'] = time.monotonic()-started
        write(output/'SUPERVISOR.json', report)
    def stop(*_):
        raise InterruptedError('Supervisor interrupted; preserve one attempt without retry')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    save()
    try:
        lock = (E/'analysis_outputs/selected_VALID_collection_gpu_slot.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['exclusive_shared_GPU_slot_held'] = True
        helper_folder = E/'qualification/portablev4_link_queued_parent_authority_repair_v1'
        sys.path.insert(0, str(helper_folder))
        helper = module(helper_folder/'supervise.py', '_remaining_resource_helper')
        peers = admission['peers']
        expected_gru = {'pid': 640428, 'start_ticks': '6055039165',
                        'peak_host_RSS_bytes': 16*GiB, 'peak_GPU_bytes': 32*GiB}
        require(peers in ([], [expected_gru]), 'Only the existing exact GRU peer or no peer is admissible')
        old = helper.process(expected_gru['pid'])
        require((old is None and not peers) or (old is not None and peers == [expected_gru]
                and old['start_ticks'] == expected_gru['start_ticks']), 'GRU identity changed; fresh root admission required')
        resources = helper.preflight({'device': 'cpu', 'peers': peers}, {
            'minimum_host_headroom_after_peer_growth_bytes': cap['host_RSS_reservation_bytes'],
            'host_RSS_reservation_bytes': cap['host_RSS_reservation_bytes'],
            'disk_reservation_bytes': cap['disk_reservation_bytes']})
        state = helper.gpu_state()
        growth, _ = gpu_budget(state, peers, [], cap['GPU_reservation_bytes'])
        resources.update(GPU=state, GPU_peer_growth_bytes=growth, resources=cap,
            host_only_metadata_preflight_not_CPU_execution=True, case=case, sole_physical_GPU=GPU)
        write(output/'RESOURCES.json', resources)
        python = str(ROOT/'.venv/bin/python')
        require(sys.executable == python, 'Run supervisor with the original qualified interpreter')
        review = inputs['python_review']
        probe = 'import sys,json;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,version=sys.version)))'
        interpreter = json.loads(subprocess.check_output([python, '-B', '-c', probe], text=True, timeout=15))
        require(interpreter == read(verify(review))['runtime'], 'Original qualified interpreter differs')
        report.update(resources=resources, interpreter=interpreter)
        if case == 'bound_observer_molbace':
            request = {'schema': 'remaining-runtime-native-child-admission-v1', 'status': 'ADMITTED', 'case': case,
                'source_manifest_sha256': admission['source_manifest_sha256'], 'resources': cap,
                'output': str(output/'native'), 'python_executable': python, 'peer_resource_review': receipt(output/'RESOURCES.json'),
                'TRAIN_updates': 38, 'TEST_calls': 0, 'automatic_retries': 0, 'exclusive_shared_GPU_slot_held': True}
            command = [python, '-B', '-u', str(HERE/'bound_observer.py'), '--admission', str(output/'CHILD_ADMISSION.json')]
        else:
            package = E/'qualification/composed_BUDDY_selected_VALID_v2'
            request = read(package/'ROOT_ADMISSION_TEMPLATE.json')
            request.update(status='ADMITTED', harness_manifest_sha256=receipt(package/'MANIFEST.json')['sha256'],
                INPUTS_sha256=receipt(package/'INPUTS.json')['sha256'], output=str(output/'native'),
                python_executable=python, qualified_python_review=review, peer_resource_review=receipt(output/'RESOURCES.json'))
            command = [python, '-B', '-u', str(package/'replay.py'), '--root', str(ROOT), '--admission', str(output/'CHILD_ADMISSION.json')]
        write(output/'CHILD_ADMISSION.json', request)
        require('torch' not in sys.modules and 'numpy' not in sys.modules, 'Supervisor must remain stdlib metadata-only')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        if case == 'bound_observer_molbace':
            host_config = E/'server51_campaign_port_v1/hosts/allocation_gpu0.json'
            env.update(BENCHMARK_HOST_CONFIG=str(host_config), BENCHMARK_HOST_CONFIG_SHA256=receipt(host_config)['sha256'])
        with (output/'child.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        launched = time.monotonic()
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(cap['supervisor_wall_seconds'])
        report.update(status='running', native_child_started=True, child=helper.process(child.pid), command=command,
                      child_admission=receipt(output/'CHILD_ADMISSION.json'))
        save()
        while child.poll() is None:
            require(time.monotonic()-launched < cap['supervisor_wall_seconds'], 'Outer wall deadline exceeded')
            own = own_processes(child.pid, helper.process)
            rss = sum(p['rss_bytes'] for p in own)
            peak_rss = max(peak_rss, rss)
            require(rss <= cap['host_RSS_reservation_bytes'], 'Own process group exceeded host RSS reservation')
            for peer in peers:
                current = helper.process(peer['pid'])
                if current is not None:
                    require(current['start_ticks'] == peer['start_ticks']
                            and current['rss_bytes'] <= peer['peak_host_RSS_bytes'], 'Peer identity/host growth envelope changed')
            try:
                state = helper.gpu_state()
            except ValueError as error:
                if str(error) == 'GPU process could not be authenticated' and child.poll() is not None:
                    break
                raise
            _, gpu = gpu_budget(state, peers, own, cap['GPU_reservation_bytes'])
            peak_gpu = max(peak_gpu, gpu)
            require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= cap['disk_reservation_bytes'],
                    'Own disk reservation exceeded')
            report.update(peak_RSS_sampled_bytes=peak_rss, peak_GPU_sampled_bytes=peak_gpu,
                          child_wall_seconds=time.monotonic()-launched)
            save()
            try:
                child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
        native = read(output/'native/RESULT.json')
        require(child.returncode == 0 and native.get('status') == 'passed' and native.get('TEST_accessed') is False
                and native.get('TRAIN_updates') == admitted_updates, 'Child failed or incomplete; preserve without retry')
        own_source(admission['source_manifest_sha256'])
        report.update(status='passed', native_result=receipt(output/'native/RESULT.json'))
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                      'traceback': traceback.format_exc(limit=12)})
    finally:
        signal.alarm(0)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                deadline = time.monotonic()+10
                while True:
                    child.poll()
                    os.killpg(child.pid, 0)
                    if time.monotonic() >= deadline:
                        os.killpg(child.pid, signal.SIGKILL)
                        report['forced_KILL_after_TERM'] = True
                        break
                    time.sleep(.1)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                report.update(status='failed', cleanup_incomplete=True)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        disk = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        cost = {'wall_seconds': time.monotonic()-started,
            'child_wall_seconds': None if launched is None else time.monotonic()-launched,
            'own_output_disk_bytes': disk, 'peak_RSS_sampled_bytes': peak_rss, 'peak_GPU_sampled_bytes': peak_gpu,
            'children_peak_RSS_bytes': usage.ru_maxrss*1024, 'children_user_seconds': usage.ru_utime,
            'children_system_seconds': usage.ru_stime, 'scope': 'own process group samples; child OS cost includes metadata/GPU probes'}
        if disk > cap['disk_reservation_bytes'] or usage.ru_maxrss*1024 > cap['host_RSS_reservation_bytes']:
            report.update(status='failed', terminal_resource_violation=True)
        report.update(cost=cost, exit_code=None if child is None else child.returncode,
            unrecorded_inflight_work_possible=bool(child is not None and report['status'] != 'passed'))
        terminal = output/'native/RESULT.json'
        if terminal.exists():
            report['native_result'] = receipt(terminal)
            report['last_native_result'] = read(terminal)
        write(output/'COST.json', cost)
        save()
        if lock is not None:
            lock.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    result = execute(args.admission)
    print(json.dumps({k: result.get(k) for k in ('status', 'case', 'native_child_started', 'failure', 'cost')}))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
