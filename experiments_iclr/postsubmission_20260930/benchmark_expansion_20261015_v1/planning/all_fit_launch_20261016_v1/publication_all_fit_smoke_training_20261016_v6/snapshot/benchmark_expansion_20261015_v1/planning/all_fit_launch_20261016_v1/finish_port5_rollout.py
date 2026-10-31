"""Finish the verified repair rollout, draining old children before new tmux owners."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import hashlib
import json
import subprocess
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
sys.path[:0] = [str(E), str(E / 'planning/three_gpu_sync_20261015_v1')]
import remote_control
import remote51


def record(name, value):
    path = HERE / name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
    return value


def invoke(script, *args):
    result = subprocess.run([sys.executable, '-B', str(HERE / script), *args],
                            capture_output=True, text=True, timeout=2400)
    record('PORT5_ROLLOUT_' + script.replace('.py', '') + ('_' + args[0] if args else '') + '.json',
           {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        raise RuntimeError(script + ': ' + result.stderr[-4000:] + result.stdout[-4000:])
    print(json.dumps({'step': script, 'args': args, 'status': 'PASS'}), flush=True)


def remote_state(host, stop=False):
    remote = remote_control if host == 'allocation' else remote51
    root = remote.REMOTE_ROOT if host == 'allocation' else remote.ROOT
    code = r'''
from pathlib import Path
import subprocess,json,os,hashlib
root=Path(ROOT_VALUE);host=HOST_VALUE;e=root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
rows=[]
for worker in ([host+'/gpu0'] if host=='allocation' else [host+'/gpu0',host+'/gpu1']):
 key=worker.replace('/','_');plan=e/'campaign_execution_20261016_v5'/('PLAN-'+key+'.json')
 value=json.loads(plan.read_text());assert value['worker']==worker and value['project_root']==str(root)
 state=Path(value['state_root']);config=e/'server51_campaign_port_v4/hosts'/(key+'.json')
 if STOP_VALUE:
  env=dict(os.environ,BENCHMARK_HOST_CONFIG=str(config),BENCHMARK_HOST_CONFIG_SHA256=hashlib.sha256(config.read_bytes()).hexdigest(),CUDA_VISIBLE_DEVICES=json.loads(config.read_text())['physical_gpu_uuid'])
  r=subprocess.run([str(root/'.venv/bin/python'),'-B',str(e/'campaign_execution_20261016_v5/launch.py'),'stop','--worker',worker,'--plan',str(plan),'--plan-sha256',hashlib.sha256(plan.read_bytes()).hexdigest()],capture_output=True,text=True,env=env)
  assert r.returncode==0,r.stderr
 status=json.loads((state/'STATUS.json').read_text())
 panes=subprocess.check_output(['tmux','list-panes','-t','gnnm-allfit-'+worker.replace('/','-')+'-20261016','-F','#{pane_dead}'],text=True).splitlines()
 rows.append({'worker':worker,'phase':status['phase'],'counts':status['counts'],'active_fit_ids':status['active_fit_ids'],'drained':bool(panes) and all(p=='1' for p in panes) and not status['active_fit_ids']})
print(json.dumps({'host':host,'workers':rows,'children_signalled':False}))
'''.replace('ROOT_VALUE', repr(root)).replace('HOST_VALUE', repr(host)).replace('STOP_VALUE', repr(stop))
    transport = 'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(base64.b64encode(zlib.compress(code.encode())).decode()) + ')))'
    assert len(transport.encode()) < 5000
    return json.loads(remote.call(transport, timeout=90))


def main():
    deadline = time.monotonic() + 1800
    projection = HERE / 'PORT5_PROJECTION_RUN_01.json'
    while not projection.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Port5 projection did not finish; old queues left running')
        time.sleep(10)
    assert json.loads(projection.read_text())['returncode'] == 0
    for host in ('allocation', 'server51'):
        assert (HERE / ('PORT5_RUNTIME_STAGE_' + host + '.json')).exists()
    diagnostic = json.loads((HERE / 'BUDDY_POLICY_DIAGNOSTIC_OBSERVATION.json').read_text())
    assert diagnostic['result']['passed'] and diagnostic['native_result']['TRAIN_updates'] == 2
    assert diagnostic['native_result']['checkpoint_reload']['status'] == 'PASS'
    invoke('deploy_port5_generated.py')
    with ThreadPoolExecutor(max_workers=2) as pool:
        drained = list(pool.map(lambda host: remote_state(host, True), ('allocation', 'server51')))
    record('PORT5_OLD_QUEUES_GRACEFUL_STOP.json', drained)
    deadline = time.monotonic() + 2100
    while not all(row['drained'] for host in drained for row in host['workers']):
        if time.monotonic() > deadline:
            raise TimeoutError('Old children have not drained; no forced interruption')
        time.sleep(10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            drained = list(pool.map(remote_state, ('allocation', 'server51')))
    record('PORT5_OLD_QUEUES_DRAINED.json', drained)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda host: invoke('prepare_launch_port5_host.py', host), ('allocation', 'server51')))
    invoke('bind_global_smoke_port5.py')
    binding = HERE / 'GLOBAL_SMOKE_BINDINGS_PORT5.json'
    result = subprocess.run([sys.executable, '-B', str(HERE / 'collect_global_smoke_barrier_port5.py'),
        '--bindings', str(binding), '--bindings-sha256', hashlib.sha256(binding.read_bytes()).hexdigest()],
        capture_output=True, text=True, timeout=180)
    record('GLOBAL_SMOKE_PORT5_PENDING_PROBE_01.json',
           {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, result.stderr
    record('PORT5_ROLLOUT_COMPLETE.json', {'queues_launched': 3,
        'prior_attempts_preserved': True, 'full_training_claimed': False,
        'binding_sha256': hashlib.sha256(binding.read_bytes()).hexdigest()})
    print('Port5 queue rollout completed; actual smoke coverage remains required.', flush=True)


if __name__ == '__main__':
    main()
