"""Launch the frozen two-update regression after server51 GPU1 drains."""
from pathlib import Path
import base64
import json
import sys
import zlib

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
sys.path.insert(0, str(E / 'planning/three_gpu_sync_20261015_v1'))
import remote51

code = r'''
from pathlib import Path
import subprocess,json,hashlib,shlex,time
root=Path(ROOT_VALUE);e=root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1';l=e/'planning/all_fit_launch_20261016_v1'
state=e/'campaign_execution_20261016_v5/runs/server51_gpu1'
s=json.loads((state/'STATUS.json').read_text());assert s['phase']=='stopped' and not s['active_fit_ids']
panes=subprocess.check_output(['tmux','list-panes','-t','gnnm-allfit-server51-gpu1-20261016','-F','#{pane_dead}'],text=True).splitlines()
assert panes and all(p=='1' for p in panes),'Ordinary queue has not drained'
package=l/'buddy_policy_port5_diagnostic_20261016_v1';source_sha='fccdb9348aa0f61f2db1609ed6dc290b96abf3218c2f9a5ca7eb0d25316a9190'
assert hashlib.sha256((package/'MANIFEST.json').read_bytes()).hexdigest()==source_sha
assert not (package/'run').exists(),'Preserve prior diagnostic'
plan=e/'campaign_execution_20261016_v5/PLAN-server51_gpu1.json';plan_sha='13197fc3c785fcfb50870e7ed662edf7e03d883ce39eb673aefab3cb615f3129'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==plan_sha
config=e/'server51_campaign_port_v4/hosts/server51_gpu1.json'
session='gnnm-buddy-policy-port5-diagnostic-gpu1-20261016'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
log=package/('OWNER-'+str(time.time_ns())+'.log')
argv=['env','BENCHMARK_HOST_CONFIG='+str(config),'BENCHMARK_HOST_CONFIG_SHA256='+hashlib.sha256(config.read_bytes()).hexdigest(),
 'CUDA_VISIBLE_DEVICES=GPU-30283973-140d-9e7b-e780-0fcc487dec54','PYTHONDONTWRITEBYTECODE=1',
 str(root/'.venv/bin/python'),'-B','-u',str(package/'run_diagnostic.py'),
 '--parent-plan',str(plan),'--parent-plan-sha256',plan_sha,'--source-sha256',source_sha]
subprocess.check_call(['tmux','new-session','-d','-s',session,'-c',str(root)])
subprocess.check_call(['tmux','set-option','-t',session,'remain-on-exit','on'])
subprocess.check_call(['tmux','send-keys','-t',session,'-l','exec '+shlex.join(argv)+' >> '+shlex.quote(str(log))+' 2>&1'])
subprocess.check_call(['tmux','send-keys','-t',session,'Enter'])
out={'session':session,'argv':argv,'owner_log':str(log),'source_sha256':source_sha,
 'diagnostic_only':True,'global_smoke_coverage_eligible':False,'full_training_launched':False}
with (l/'BUDDY_POLICY_DIAGNOSTIC_LAUNCH.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
'''.replace('ROOT_VALUE', repr(remote51.ROOT))
transport = 'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(base64.b64encode(zlib.compress(code.encode())).decode()) + ')))'
assert len(transport.encode()) < 5000
out = json.loads(remote51.call(transport, timeout=60))
(HERE / 'BUDDY_POLICY_DIAGNOSTIC_LAUNCH.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
