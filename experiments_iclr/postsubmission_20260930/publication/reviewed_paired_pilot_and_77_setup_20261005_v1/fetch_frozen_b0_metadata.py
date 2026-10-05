#!/usr/bin/env python3
"""Read only exact frozen prospective b0 metadata on the authorized singleton."""
import hashlib,json,shlex,subprocess
from pathlib import Path

PHASE=Path(__file__).resolve().parents[2]
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
DONOR='shared_private_transfer_paired_pilot_execution_root_20261005_v2'
REMOTE=r'''
import hashlib,json,pathlib,socket,subprocess,sys
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];donor=sys.argv[3]
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()==[uuid]
phase=repo/'experiments_iclr/postsubmission_20260930';root=phase/donor
assert root.resolve().is_relative_to(phase.resolve())
paths=['COHORT_PLAN.json','QUEUE.json','ROOT_RELEASE.json','EXTERNAL_ANCHORS.json']
plan=json.loads((root/'COHORT_PLAN.json').read_text())
assert len(plan['cells'])==30 and len({row['cell_id'] for row in plan['cells']})==30
paths += ['jobs/'+row['cell_id']+'.json' for row in plan['cells']]
rows=[]
for relative in paths:
 path=root/relative;assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
 data=path.read_bytes();assert len(data)<1000000
 rows.append({'relative':donor+'/'+relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'utf8':data.decode()})
print(json.dumps({'hostname':socket.gethostname(),'GPU_UUID':uuid,'scope':'frozen prospective metadata only','scores_read':False,'models_or_TEST_access':False,'files':rows}))
'''

def main():
    dest=Path(__file__).resolve().parent
    if (dest/'METADATA_FETCH_RECEIPT.json').exists():raise ValueError('Preserve prior retrieval; no implicit rerun')
    receipt=PHASE/'shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/LAUNCH_RECEIPT.json'
    authority=json.loads(receipt.read_text())
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO,UUID,DONOR])
    result=subprocess.run([*ssh,command],capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError('Exact metadata retrieval failed: '+result.stderr)
    value=json.loads(result.stdout);rows=value['files'];lookup={r['relative']:r for r in rows}
    for filename,key in [('COHORT_PLAN.json','cohort_plan_sha256'),('QUEUE.json','queue_sha256'),('ROOT_RELEASE.json','release_sha256')]:
        if lookup[DONOR+'/'+filename]['sha256']!=authority[key]:raise ValueError('Metadata differs from immutable launch authority')
    for row in rows:
        path=PHASE/row['relative'];data=row['utf8'].encode()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Retrieved metadata custody differs')
        if path.exists() and path.read_bytes()!=data:raise ValueError('Never overwrite different local metadata')
    for row in rows:
        path=PHASE/row['relative'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_text(row['utf8'])
    evidence={key:v for key,v in value.items() if key!='files'}
    evidence.update(ssh_destination=LOGIN,launch_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        fetch_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files=[{key:v for key,v in row.items() if key!='utf8'} for row in rows])
    (dest/'METADATA_FETCH_RECEIPT.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print('Retrieved and authenticated '+str(len(rows))+' prospective JSON files; no scientific outputs accessed.')


if __name__=='__main__':main()
