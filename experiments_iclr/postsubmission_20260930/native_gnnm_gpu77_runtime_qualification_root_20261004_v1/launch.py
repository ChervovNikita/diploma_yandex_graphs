from remote_transport import run,HERE,REPO,PHASE,PYTHON
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
REMOTE=PHASE/HERE.name
UUID='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
UUIDS=[UUID,'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
rows=json.loads((HERE/'STAGE_INVENTORY.json').read_text())
code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport json,os,hashlib,subprocess\n'
code+='repo=Path('+repr(str(REPO))+');root=Path('+repr(str(REMOTE))+');python='+repr(PYTHON)+';uuid='+repr(UUID)+';uuids='+repr(UUIDS)+'\n'
code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code+='assert subprocess.run(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.splitlines()==uuids\n'
code+='rows='+repr(rows)+'\n'
code+='for r in rows:\n f=root/r["path"];assert f.resolve().is_relative_to(root) and not f.is_symlink();assert f.stat().st_size==r["bytes"] and hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"]\n'
code+='assert not (root/"DETACHED_LAUNCH.json").exists() and not (root/"QUALIFICATION.json").exists()\n'
code+='assert not (root/"DETACHED.stdout").exists() and not (root/"DETACHED.stderr").exists()\n'
code+='free=int(subprocess.run(["nvidia-smi","--id="+uuid,"--query-gpu=memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.strip());assert free>=35000\n'
code+='command=[python,"-B",str(root/"supervisor.py")]\n'
code+='with (root/"DETACHED.stdout").open("xb") as out,(root/"DETACHED.stderr").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
code+='raw=(Path("/proc")/str(child.pid)/"stat").read_text();fields=raw[raw.rfind(")")+2:].split()\n'
code+='value=dict(UTC=datetime.now(timezone.utc).isoformat(),status="TWO_TRAIN_ONLY_NATIVE_GNNM_COMPONENT_CASES_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]),command=command,physical_UUIDs=uuids,selected_GPU_UUID=uuid,free_MiB_at_start=free,source_payloads_verified=len(rows),automatic_retry=False,job_signals_sent=False,VALIDATION_or_TEST_targets_read=False,scientific_training_updates=0)\n'
code+='with (root/"DETACHED_LAUNCH.json").open("x") as stream:json.dump(value,stream,indent=2);stream.write("\\n")\nprint(json.dumps(value))\n'
(HERE/'REMOTE_LAUNCH.py.txt').write_text(code)
value=run('native_gnnm_gpu77_runtime_launch_20261004_v1',code)
with (HERE/'DETACHED_LAUNCH.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
