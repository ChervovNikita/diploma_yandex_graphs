"""Observe the two existing owned queues without copying old fit artifacts."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
TRANSPORT = PHASE / "ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py"
spec = importlib.util.spec_from_file_location("existing_maclink_transport", TRANSPORT)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.HERE = HERE

cb_local = PHASE / "exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1"
ddi_local = PHASE / "ddi_paired_development_pilot_execution_root_20261004_v1"
cb_identity = json.loads((cb_local / "DETACHED_LAUNCH.json").read_text())["queue_physical_identity"]
ddi_identity = json.loads((ddi_local / "DETACHED_LAUNCH.json").read_text())

CODE = '''from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo and os.uname().nodename=='peptide'
g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
assert set(g.stdout.splitlines())=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
def read(root,name):
 p=root/name
 if not p.exists():return None
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<2000000
 return json.loads(p.read_text())
def physical(expected):
 pid=expected.get('PID',expected.get('pid'))
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  actual=dict(PID=pid,start_ticks=int(f[19]),state=f[0],cwd=str((p/'cwd').resolve()),argv=argv)
 except FileNotFoundError:return None
 ticks=expected.get('start_time_ticks',expected.get('start_ticks'))
 assert actual['start_ticks']==ticks
 if actual['state']!='Z':
  assert actual['cwd']==str(repo)
  if 'argv' in expected:assert argv==expected['argv']
  elif 'command' in expected:assert argv==expected['command']
 return actual
cb=phase/'exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1'
ddi=phase/'ddi_paired_development_pilot_execution_root_20261004_v1'
cstatus=read(cb,'queue/run01/STATUS.json')
c=dict(queue_handle=physical(CB_IDENTITY),progress=cstatus,terminal=read(cb,'queue/run01/TERMINAL.json'))
if cstatus and cstatus.get('active_cell'):
 name=cstatus['active_cell'];assert name in CB_CELLS
 s=read(cb,name+'/run01/STATUS.json')
 if s is not None:
  assert s['predictive_values_exposed'] is False
  c['current_fit']={k:s[k] for k in ('status','phase','work','observed_wall_seconds')}
 child=read(cb,'supervision/'+name+'/run01/CHILD_STARTED.json')
 c['current_child']=physical(child['physical_identity']) if child else None
dstatus=read(ddi,'QUEUE_PROGRESS.json')
d=dict(queue_handle=physical(DDI_IDENTITY),progress=dstatus,terminal=read(ddi,'QUEUE_TERMINAL.json'),exception=read(ddi,'QUEUE_EXCEPTION.json'))
if dstatus and dstatus.get('current_cell'):
 cell=dstatus['current_cell']['cell_id'];assert cell in DDI_CELLS
 child=read(ddi,'supervision/'+cell+'/OWNED_CHILD.json')
 d['current_child']=physical(child['identity']) if child else None
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=dict(repository=str(repo),hostname=os.uname().nodename,GPU_UUIDs=g.stdout.splitlines()),collab=c,ddi=d,metadata_only=True,predictive_values_read=False,signals_sent=False,retries=False)))
'''
cb_cells = [c["cell"] for c in json.loads((cb_local / "QUEUE_RELEASE.json").read_text())["queue_order"]]
ddi_cells = [c["cell_id"] for c in json.loads((ddi_local / "PLAN.json").read_text())["queue"]]
CODE = "CB_IDENTITY=" + repr(cb_identity) + "\nDDI_IDENTITY=" + repr(ddi_identity) + "\nCB_CELLS=" + repr(cb_cells) + "\nDDI_CELLS=" + repr(ddi_cells) + "\n" + CODE

if __name__ == "__main__":
    with (HERE / "REMOTE_CODE.py.txt").open("x") as handle:
        handle.write(CODE)
    result = transport.run("compact_owned_77_queues_20261005_v1", CODE)
    with (HERE / "OBSERVATION.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
