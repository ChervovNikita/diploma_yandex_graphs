"""Prepare one ordinary 18.77 evaluation command for the complete six-fit study."""
from pathlib import Path
import hashlib
import json
import shlex

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
WRAPPER = PHASE / 'gpu77_connection_recovery_v1'

def descriptor(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(PHASE)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

release = json.loads((PHASE / 'amazon_ratings_native_warm_study_preparation_20261003_v2/EVALUATION_RELEASE_TEMPLATE.json').read_text())
training = json.loads((HERE / 'TRAINING_RELEASE.json').read_text())
for key in ('packet_manifest', 'source_review', 'runtime_receipt', 'data_manifest'):
    release[key] = training[key]
    assert descriptor(PHASE / training[key]['path']) == training[key]
release.update(execution_authorized=True, root_observed_source_review=True,
               output=HERE.name + '/evaluation_v1', study=descriptor(HERE / 'training/STUDY.json'),
               root_observed_training_process_exit=True, root_observed_training_exit_code=0)
inputs = [descriptor(HERE / name) for name in ('TRAINING_DISPATCH.json', 'TRAINING_CONSOLE_TERMINAL.json')]
source = '''from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, subprocess, shlex
repo = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase = repo / 'experiments_iclr/postsubmission_20260930'
root = phase / 'amazon_ratings_native_warm_execution_root_20261003_v3'
os.chdir(repo)
assert subprocess.run(['git','rev-parse','--show-toplevel'], capture_output=True, text=True, check=True).stdout.strip() == str(repo)
assert set(subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.splitlines()) == {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
def descriptor(path):
    b = path.read_bytes()
    return {'path':str(path.relative_to(phase)), 'sha256':hashlib.sha256(b).hexdigest(), 'bytes':len(b)}
def write(path, value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\\n')
release = json.loads(RELEASE_JSON)
inputs = json.loads(INPUTS_JSON)
for entry in inputs + [release[k] for k in ('packet_manifest','source_review','runtime_receipt','data_manifest','study')]:
    assert descriptor(phase / entry['path']) == entry
dispatch = json.loads((root / 'TRAINING_DISPATCH.json').read_text())
terminal = json.loads((root / 'TRAINING_CONSOLE_TERMINAL.json').read_text())
study = json.loads((root / 'training/STUDY.json').read_text())
assert terminal['exit_code'] == 0 and study['all6_closed'] and study['all6_selected_replayed'] and len(study['rows']) == 6
assert all(row['status'] == 'selected' and row['completed'] for row in study['rows'])
assert not study['TEST_labels_used_or_scored'] and dispatch['console_PID'] == 3139323
assert dispatch['starttime_ticks'] == '1717959955'
process_observations = []
for pid in (3139323,3139324):
    stat = Path('/proc') / str(pid) / 'stat'
    if stat.exists():
        text = stat.read_text();fields = text[text.rfind(')')+2:].split()
        assert fields[19] != '1717959955', 'Original training process remains present'
        process_observations.append({'PID':pid,'present_reused_start_ticks':fields[19],'original_training_process_absent':True})
    else:
        process_observations.append({'PID':pid,'present':False,'original_training_process_absent':True})
assert not (root / 'evaluation_v1').exists()
observation = {'schema':'root-complete6-training-process-exit-observation-v1',
               'UTC':datetime.now(timezone.utc).isoformat(), 'training_dispatch':inputs[0],
               'console_terminal':inputs[1], 'study':release['study'], 'exit_code':0,
               'processes':process_observations,'all6_closed':True,'all6_selected_replayed':True,
               'TEST_labels_used_or_scored':False,'no_jobs_restarted_or_stopped':True}
write(root / 'TRAINING_PROCESS_EXIT_OBSERVATION_v1.json', observation)
release['training_process_exit_receipt'] = descriptor(root / 'TRAINING_PROCESS_EXIT_OBSERVATION_v1.json')
release_path = root / 'EVALUATION_RELEASE_v1.json'
write(release_path, release)
env = os.environ.copy()
env.update(PYTHONPATH=str(repo / '.gnnm_runtime/conditional_pyg27_v1/site') + ':' + str(repo / '.gnnm_runtime/conditional_xxhash_v1/site'),
           CUDA_VISIBLE_DEVICES='GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
argv = ['/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12','-B',
        str(phase / 'amazon_ratings_native_warm_study_preparation_20261003_v2/evaluate_study.py'),
        '--admission',str(release_path),'--data',str(root / 'data/DATA_MANIFEST.json'),
        '--study',str(root / 'training/STUDY.json'),'--device','cuda:0','--output',str(root / 'evaluation_v1')]
exit_writer = "from pathlib import Path;from datetime import datetime,timezone;import json,sys;p=Path(sys.argv[1]);f=p.open('x');json.dump({'schema':'normal-console-process-exit-v1','UTC':datetime.now(timezone.utc).isoformat(),'exit_code':int(sys.argv[2]),'automatic_restart':False,'isolation_used':False},f,indent=2);f.close()"
shell = shlex.join(argv) + '; evaluation_exit_code=$?; ' + shlex.join(['/usr/bin/python3','-B','-c',exit_writer,str(root / 'EVALUATION_CONSOLE_TERMINAL_v1.json')]) + ' "$evaluation_exit_code"; exit "$evaluation_exit_code"'
with (root / 'EVALUATION_CONSOLE_v1.log').open('xb') as log:
    process = subprocess.Popen(['/bin/sh','-c',shell],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
text = (Path('/proc') / str(process.pid) / 'stat').read_text()
record = {'schema':'amazon-evaluation-normal-console-dispatch-v1','UTC':datetime.now(timezone.utc).isoformat(),
          'console_PID':process.pid,'starttime_ticks':text[text.rfind(')')+2:].split()[19],
          'argv':argv,'shell_command':shell,'GPU_UUID':env['CUDA_VISIBLE_DEVICES'],
          'release':descriptor(release_path),'process_exit_observation':release['training_process_exit_receipt'],
          'ordinary_host_execution':True,'isolation_used':False,'no_kill_restart_or_automatic_retry':True,'other_jobs_mutated':False}
write(root / 'EVALUATION_DISPATCH_v1.json',record)
print(json.dumps(record))
'''
source = source.replace('RELEASE_JSON', repr(json.dumps(release))).replace('INPUTS_JSON', repr(json.dumps(inputs)))
compile(source, '<remote Amazon evaluation dispatch>', 'exec')
command = shlex.join(['/usr/bin/python3','-B','-c',source]) + '\n'
with (WRAPPER / 'AMAZON77_COMPLETE6_EVALUATION_DISPATCH_COMMAND_v1.txt').open('x') as handle:
    handle.write(command)
print(json.dumps({'command_sha256':hashlib.sha256(command.encode()).hexdigest(),'study':release['study'],'no_numerical_execution':True}))
