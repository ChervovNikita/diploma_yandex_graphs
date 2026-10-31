"""Stage the frozen smoke-only scheduler and resume existing server51 plans."""
from pathlib import Path
import base64
import hashlib
import json
import sys
import zlib

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
sys.path[:0] = [str(HERE), str(E / "planning/three_gpu_sync_20261015_v1")]
import stage_batch
import remote51

package = E / "campaign_smoke_continuation_20261016_v1"
manifest = package / "MANIFEST.json"
source_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert source_sha == sys.argv[1], "Expected the reviewed source pin"
paths = [manifest]
for row in json.loads(manifest.read_text())["files"]:
    path = package / row["path"]
    raw = path.read_bytes()
    assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
    paths.append(path)
stage = stage_batch.stage("server51", paths)
(HERE / "SERVER51_SMOKE_CONTINUATION_STAGE.json").write_text(json.dumps(stage, indent=2) + "\n")
launch = json.loads((HERE / "TMUX_SMOKE5_LAUNCH_server51.json").read_text())
plans = [{"worker": row["worker"], "plan": row["launch"]["plan"]} for row in launch["workers"]]
code = r'''
from pathlib import Path
import hashlib,json,subprocess,shlex,time
root=Path(ROOT_VALUE);e=root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
package=e/'campaign_smoke_continuation_20261016_v1';source_sha=SHA_VALUE
assert hashlib.sha256((package/'MANIFEST.json').read_bytes()).hexdigest()==source_sha
results=[]
for row in PLANS_VALUE:
 worker=row['worker'];plan=row['plan'];path=Path(plan['path'])
 raw=path.read_bytes();assert len(raw)==plan['bytes'] and hashlib.sha256(raw).hexdigest()==plan['sha256']
 value=json.loads(raw);state=Path(value['state_root']);status=json.loads((state/'STATUS.json').read_text())
 assert status['phase']=='smoke_failed' and not status['active_fit_ids'],status['phase']
 session='gnnm-allfit-'+worker.replace('/','-')+'-20261016'
 panes=subprocess.check_output(['tmux','list-panes','-t',session,'-F','#{pane_dead}'],text=True).splitlines()
 assert panes and all(p=='1' for p in panes),'Refuse to replace live tmux'
 config=e/'server51_campaign_port_v4/hosts'/(worker.replace('/','_')+'.json')
 cfg=json.loads(config.read_text());config_sha=hashlib.sha256(config.read_bytes()).hexdigest()
 log=state/('CONTINUATION-'+str(time.time_ns())+'.log')
 argv=['env','BENCHMARK_HOST_CONFIG='+str(config),'BENCHMARK_HOST_CONFIG_SHA256='+config_sha,
       'CUDA_VISIBLE_DEVICES='+cfg['physical_gpu_uuid'],'PYTHONDONTWRITEBYTECODE=1',
       str(root/'.venv/bin/python'),'-B','-u',str(package/'continue_smokes.py'),
       '--worker',worker,'--plan',str(path),'--plan-sha256',plan['sha256'],'--source-sha256',source_sha]
 subprocess.check_call(['tmux','kill-session','-t',session])
 subprocess.check_call(['tmux','new-session','-d','-s',session,'-c',str(root)])
 subprocess.check_call(['tmux','set-option','-t',session,'remain-on-exit','on'])
 command='exec '+shlex.join(argv)+' >> '+shlex.quote(str(log))+' 2>&1'
 subprocess.check_call(['tmux','send-keys','-t',session,'-l',command])
 subprocess.check_call(['tmux','send-keys','-t',session,'Enter'])
 results.append({'worker':worker,'session':session,'plan':plan,'source_sha256':source_sha,
                 'owner_log':str(log),'full_training_supported':False})
out={'workers':results,'failed_fits_retried':False,'prior_evidence_preserved':True}
p=e/'planning/all_fit_launch_20261016_v1/SERVER51_SMOKE_CONTINUATION_LAUNCH.json'
with p.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
'''.replace("ROOT_VALUE", repr(remote51.ROOT)).replace("SHA_VALUE", repr(source_sha)).replace("PLANS_VALUE", repr(plans))
transport = "import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(base64.b64encode(zlib.compress(code.encode())).decode()) + ")))"
assert len(transport.encode()) < 5000
out = json.loads(remote51.call(transport, timeout=90))
(HERE / "SERVER51_SMOKE_CONTINUATION_LAUNCH.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
