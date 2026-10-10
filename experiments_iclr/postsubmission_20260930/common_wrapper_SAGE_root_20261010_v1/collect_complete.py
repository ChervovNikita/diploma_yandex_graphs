"""Fetch only bounded complete-family metadata after the finite owner closes."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import base64,hashlib,json,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
h=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/common_wrapper_SAGE_root_20261010_v1')
end=json.loads((h/'OWNER_END.json').read_text());launch=json.loads((h/'LAUNCH.json').read_text())
assert end['scientific_success'] and end['direct_child_wait'] and end['child_pid_absent'] and end['owned_cuda_pid_absent']
assert not Path('/proc',str(end['child']['PID'])).exists()
assert not Path('/proc',str(launch['PID'])).exists()
q=h/'actual_family_v1/COMPLETE_FAMILY.json';assert q.stat().st_size<500000
raw=q.read_bytes();report=json.loads(raw)
assert report['complete'] and report['groups']==21 and report['fit_units']==39 and report['TEST_access'] is False
assert hashlib.sha256(raw).hexdigest()==end['complete_sha256']
print(json.dumps(dict(complete_b64=base64.b64encode(raw).decode(),complete_sha256=end['complete_sha256'],owner_end=end,launch=launch)))
'''

def main():
    here=Path(__file__).resolve().parent
    assert not (here/'ACTUAL_COMPLETE_FAMILY.json').exists()
    cmd=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',cmd]
    r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    if r.returncode:
        print(json.dumps(dict(exit_code=r.returncode,stderr=r.stderr,stdout=r.stdout)))
        return r.returncode
    value=json.loads(r.stdout);data=base64.b64decode(value.pop('complete_b64'))
    assert hashlib.sha256(data).hexdigest()==value['complete_sha256']
    (here/'ACTUAL_COMPLETE_FAMILY.json').write_bytes(data)
    (here/'ACTUAL_COMPLETE_CUSTODY.json').write_text(json.dumps(value,indent=2)+'\n')
    report=json.loads(data)
    means={}
    for arm in sorted({x['arm'] for x in report['results']}):
        rows=[x for x in report['results'] if x['arm']==arm]
        means[arm]=dict(accuracy_percent=sum(x['valid']['accuracy'] for x in rows)/3*100,
                       nll=sum(x['valid']['nll'] for x in rows)/3,
                       seed_accuracy_percent=[x['valid']['accuracy']*100 for x in rows])
    print(json.dumps(dict(complete_sha256=value['complete_sha256'],groups=21,fit_units=39,means=means)))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
