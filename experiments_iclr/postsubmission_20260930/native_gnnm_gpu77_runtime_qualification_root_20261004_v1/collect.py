from remote_transport import run,HERE,REPO,PHASE
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,hashlib,json
parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=int,required=True);args=parser.parse_args()
REMOTE=PHASE/HERE.name
launch=json.loads((HERE/'DETACHED_LAUNCH.json').read_text())
paths=['DETACHED_LAUNCH.json','DETACHED.stdout','DETACHED.stderr','local_CHILD_STARTED.json','local_PHYSICAL_TERMINAL.json','local.stdout','local.stderr','global_CHILD_STARTED.json','global_PHYSICAL_TERMINAL.json','global.stdout','global.stderr','native_reference_ordinary_native_gnnm_local.json','native_reference_ordinary_native_gnnm_global.json','QUALIFICATION.json','SUPERVISOR_TERMINAL.json']
code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport json,os,hashlib,base64,subprocess\n'
code+='repo=Path('+repr(str(REPO))+');root=Path('+repr(str(REMOTE))+');pid='+str(launch['supervisor_PID'])+';ticks='+str(launch['supervisor_start_ticks'])+';paths='+repr(paths)+'\n'
code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code+='state={"exists":False};proc=Path("/proc")/str(pid)\n'
code+='if (proc/"stat").exists():\n raw=(proc/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[19])==ticks;state={"exists":True,"state":f[0],"start_ticks":int(f[19]),"proc_exit_code":int(f[49]) if len(f)>49 else None,"cmdline":(proc/"cmdline").read_bytes().replace(b"\\x00",b" ").decode(errors="replace")}\n'
code+='child_states={}\nfor stage in ("local","global"):\n f=root/(stage+"_CHILD_STARTED.json")\n if f.exists():\n  ident=json.loads(f.read_text())["identity"];cp=Path("/proc")/str(ident["PID"])/"stat";row={"PID":ident["PID"],"expected_start_ticks":ident["start_ticks"],"exists":cp.exists()}\n  if cp.exists():\n   raw=cp.read_text();cf=raw[raw.rfind(")")+2:].split();row.update(state=cf[0],same_identity=int(cf[19])==ident["start_ticks"])\n  child_states[stage]=row\n'
code+='rows=[]\nfor rel in paths:\n path=root/rel;assert path.resolve().is_relative_to(root) and not path.is_symlink()\n if path.is_file():\n  raw=path.read_bytes();rows.append({"path":rel,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"data":base64.b64encode(raw).decode()})\n'
code+='print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_state=state,child_states=child_states,files=rows,selected_GPU_inventory=subprocess.run(["nvidia-smi","--id=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998","--query-gpu=uuid,memory.free","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout,job_signals_sent=False)))\n'
value=run('native_gnnm_gpu77_runtime_owned_monitor%03d_20261004_v1'%args.sequence,code)
out=HERE/('owned_monitor%03d'%args.sequence);out.mkdir()
for r in value['files']:
 raw=base64.b64decode(r['data'],validate=True);assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 with (out/r['path']).open('xb') as f:f.write(raw)
 if r['path'] in ('QUALIFICATION.json','SUPERVISOR_TERMINAL.json','native_reference_ordinary_native_gnnm_local.json','native_reference_ordinary_native_gnnm_global.json'):
  target=HERE/r['path']
  if target.exists():assert target.read_bytes()==raw
  else:
   with target.open('xb') as f:f.write(raw)
summary={k:v for k,v in value.items() if k!='files'}
summary['files']=[{k:v for k,v in r.items() if k!='data'} for r in value['files']]
with (out/'MONITOR.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
if (out/'QUALIFICATION.json').is_file():
 q=json.loads((out/'QUALIFICATION.json').read_text());summary['completed']=q['all_component_checks_complete'];summary['cases']=[{'stage':r['stage'],'exit_code':r.get('exit_code'),'wall_seconds':r.get('wall_seconds'),'step_seconds':r.get('result',{}).get('step_seconds'),'function_replay':r.get('result',{}).get('function_replay'),'next_update_replay':r.get('result',{}).get('next_update_replay'),'stderr':r.get('stderr')} for r in q['cases']]
print(json.dumps(summary))
