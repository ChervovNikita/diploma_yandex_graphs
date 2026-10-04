"""Stage the sealed Pubmed bridge qualifier and inspect prerequisites only."""
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
RELAY = PHASE/'gpu77_connection_recovery_v1'
WRAPPER = RELAY/'run_gpu77_v3.py'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO/'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE/HERE.name
SOURCE = PHASE/'pubmed_shared4_bridge_qualification_source_20261004_v2'

def run(identity, code):
    command = 'cd '+shlex.quote(str(REPO))+" && /usr/bin/python3 -I -S -B - <<'PUBROOTPY'\n"+code+'\nPUBROOTPY\n'
    assert len(command.encode())<100000
    path = RELAY/(identity+'.txt')
    with path.open('x') as f:f.write(command)
    result = subprocess.run([sys.executable,'-B',str(WRAPPER),'--id',identity,'--command-file',str(path)],capture_output=True,text=True)
    with (HERE/(identity+'_LOCAL_TRANSPORT.json')).open('x') as f:
        json.dump(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,command_sha256=hashlib.sha256(command.encode()).hexdigest()),f,indent=2);f.write('\n')
    assert result.returncode==0,result.stderr+result.stdout[:1500]
    outer=json.loads(result.stdout);assert outer['exit_code']==0
    return json.loads(outer['stdout'])

def stage():
    inventory=json.loads((HERE/'SOURCE_METADATA_STAGE_INVENTORY.json').read_text())
    payload=[]
    for row in inventory:
        raw=(PHASE/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        payload.append(dict(row,data=base64.b64encode(raw).decode()))
    compressed=zlib.compress(json.dumps(payload).encode(),9)
    encoded=base64.b64encode(compressed).decode()
    chunks=[encoded[n:n+40000] for n in range(0,len(encoded),40000)]
    staging=REMOTE/'source_metadata_chunks_v2'
    receipts=[]
    for number,chunk in enumerate(chunks,1):
        code='from pathlib import Path\nimport hashlib,json,os\n'
        code+='repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(staging))+')\n'
        code+='assert Path.cwd()==repo and os.uname().nodename=="peptide" and root.resolve().is_relative_to(phase)\n'
        code+='root.mkdir(parents=True,exist_ok=True);p=root/'+repr('part%03d.b64'%number)+';b='+repr(chunk)+'.encode()\n'
        code+='with p.open("xb") as f:f.write(b)\n'
        code+='print(json.dumps(dict(status="SOURCE_METADATA_CHUNK_STAGED",bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),scientific_execution=False)))\n'
        result=run('pubmed_bridge_qual_v2_metadata_chunk%03d_20261004'%number,code)
        assert result['sha256']==hashlib.sha256(chunk.encode()).hexdigest()
        receipts.append(result)
        print(json.dumps(dict(chunk=number,total=len(chunks))),flush=True)
    source_binding=json.loads((SOURCE/'SOURCE_BINDING.json').read_text())
    code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess,zlib\n'
    code+='repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(staging))+')\n'
    code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
    code+='compressed=base64.b64decode(b"".join((root/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\n'
    code+='assert hashlib.sha256(compressed).hexdigest()=='+repr(hashlib.sha256(compressed).hexdigest())+'\nrows=json.loads(zlib.decompress(compressed))\n'
    code+="for r in rows:\n p=(phase/r['path']).resolve();assert p.is_relative_to(phase);b=base64.b64decode(r['data'],validate=True);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']\n if p.exists():assert not p.is_symlink() and p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open('xb') as f:f.write(b)\n"
    code+='specifications='+repr(source_binding['prerequisite_receipts'])+'\nprerequisites={}\n'
    code+="for name,r in specifications.items():\n p=phase/r['path'];assert p.resolve().is_relative_to(phase) and p.is_file() and not p.is_symlink();b=p.read_bytes();assert len(b)<2000000;digest=hashlib.sha256(b).hexdigest();v=json.loads(b);assert not r['sha256'] or digest==r['sha256'];assert v.get('status')==r['required_status'];prerequisites[name]=dict(path=r['path'],sha256=digest,bytes=len(b),status=v['status'],verified_by_root=True)\n"
    code+="g=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total,name','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert g.returncode==0\n"
    code+="gpu_rows=[[v.strip() for v in line.split(',')] for line in g.stdout.splitlines()];selected=next(r for r in gpu_rows if r[0]=='GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced');assert int(selected[1])>=73728 and int(selected[2])==81920 and selected[3]=='NVIDIA A100 80GB PCIe'\n"
    code+="available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'));assert available>=40*1024**3\n"
    code+="print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_EXACT_SOURCE_AND_PREREQUISITE_METADATA',files=len(rows),bytes=sum(r['bytes'] for r in rows),prerequisites=prerequisites,GPU_rows=gpu_rows,host_MemAvailable_bytes=available,study_arrays_states_or_TEST_opened=False,numerical_execution=False)))\n"
    result=run('pubmed_bridge_qual_v2_metadata_join_20261004',code)
    with (HERE/'SOURCE_PREREQUISITE_RESOURCE_STAGE_RECEIPT.json').open('x') as f:json.dump(dict(result,chunks=receipts),f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('UTC','status','files','bytes','GPU_rows','host_MemAvailable_bytes','numerical_execution')}))

if __name__=='__main__':stage()
