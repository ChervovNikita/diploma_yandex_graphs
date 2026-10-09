from pathlib import Path
import json,socket,subprocess,datetime
assert socket.gethostname()=='anogena-2-0'
U='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[U]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';Q=P/'internal_BE_molhiv18_family_execution_root_20261007_v1'
t=json.loads((Q/'TERMINAL_EVIDENCE.json').read_text());parent=json.loads((Q/'PARENT_OWNER.json').read_text());handles=list(t['handles'].values())+[parent]
own_pids={h['pid'] for h in handles};groups={h['group'] for h in handles};observed=[];members=[]
for h in handles:
 p=Path('/proc',str(h['pid']),'stat')
 try:s=p.read_text()
 except FileNotFoundError:observed.append(dict(pid=h['pid'],expected_start=h['start_ticks'],same_process_present=False));continue
 f=s[s.rfind(')')+2:].split();same=int(f[19])==h['start_ticks'];observed.append(dict(pid=h['pid'],expected_start=h['start_ticks'],same_process_present=same,observed_start=int(f[19]),state=f[0]));assert not same or f[0]=='Z'
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:s=(p/'stat').read_text()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 f=s[s.rfind(')')+2:].split()
 if int(f[2]) in groups and f[0]!='Z':members.append(dict(pid=int(p.name),group=int(f[2]),state=f[0]))
assert not members
cuda=[]
for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).splitlines():
 if row.split(',')[0].strip().isdigit() and int(row.split(',')[0]) in own_pids:cuda.append(row)
assert not cuda
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),all_owned_original_handles_absent=True,observed_handles=observed,own_group_live_members=members,owned_CUDA=cuda,GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines(),original_scores_opened=False)))