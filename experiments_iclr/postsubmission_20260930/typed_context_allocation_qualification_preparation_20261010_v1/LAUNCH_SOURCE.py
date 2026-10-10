import json,hashlib,os,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'typed_context_allocation_qualification_preparation_20261010_v1'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='24c315a7-3c08-471f-b550-b9a3e1faf75d'
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True,timeout=10).strip()=='d9a6a39bfcbc648279128cf0772f541cf2c04e97'
adopt=json.loads((A/'OWNER_ADOPTION.json').read_text());assert adopt['approved'] and not adopt['scientific']
for key in ['owner','entry','release','owner_assessment']:
 row=adopt[key];q=Path(row['path']);assert q.resolve().is_relative_to(P) and q.stat().st_size==row['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
release=json.loads((A/'RELEASE.json').read_text());assert release['enabled'] and not release['scientific_execution_approved']
assert not Path(release['output_directory']).exists()
assert not any((A/n).exists() for n in ['STARTER.json','LAUNCH.json','TERMINAL.json','owner.stdout','owner.stderr'])
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
assert free>=release['resource_budget']['device_bytes']
with (A/'owner.stdout').open('xb') as out,(A/'owner.stderr').open('xb') as err:
 parent=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(A/'OWNED_QUALIFY.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 s=Path('/proc',str(parent.pid),'stat').read_text().rsplit(')',1)[1].split()
 value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=parent.pid,start_ticks=int(s[19]),pgid=int(s[2]),sid=int(s[3]),boot_id='24c315a7-3c08-471f-b550-b9a3e1faf75d',source_commit='d9a6a39bfcbc648279128cf0772f541cf2c04e97',GPU_free_bytes=free,source_qualification_only=True,scientific=False,automatic_retry=False,other_jobs_changed=False)
 with (A/'STARTER.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write(chr(10))
 print(json.dumps(value),flush=True)
