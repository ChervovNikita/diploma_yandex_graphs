from pathlib import Path
import subprocess,socket,json,datetime,os
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_IMDB_paired_independent_reference_activation_root_20261009_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='881ce362a8a150b63ac8ba1c9c49b70b14393bdb'
assert not (A/'STARTER.json').exists() and not (A/'LAUNCH.json').exists() and not (A/'TERMINAL.json').exists()
old=P/'sehgnn_IMDB_full_input_qualification_activation_root_20261009_v2';t=json.loads((old/'TERMINAL.json').read_text());old_launch=json.loads((old/'LAUNCH.json').read_text())
assert t['exit_code']==0 and t['child_reaped'] and t['original_pid_absent']
assert not Path('/proc',str(old_launch['parent']['pid'])).exists() and not Path('/proc',str(old_launch['child']['pid'])).exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert free>=24*1024**3,'Native qualifier waits for its declared device headroom; no other job changed'
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')};assert mem['MemAvailable']>=32*1024**3,'Wait for declared host headroom; no other job changed'
with (A/'owner.stdout').open('x') as stdout,(A/'owner.stderr').open('x') as stderr:
 parent=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(A/'OWNED_RUN.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
 stat=Path('/proc',str(parent.pid),'stat').read_text();fields=stat[stat.rfind(')')+2:].split();handle=dict(pid=parent.pid,start_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=handle,source_commit='881ce362a8a150b63ac8ba1c9c49b70b14393bdb',fresh_output=True,native_math_and_recipe_unchanged=True,automatic_retry=False,other_jobs_changed=False,free_device_bytes_at_launch=free)
 receipt['declared_reference_body_fits']=24
 with (A/'STARTER.json').open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
 print(json.dumps(receipt))
