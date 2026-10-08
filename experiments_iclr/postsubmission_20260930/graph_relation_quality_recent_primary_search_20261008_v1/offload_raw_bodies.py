"""Preserve processed literature bodies in identity-checked project custody."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import stat
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE = REPO + '/experiments_iclr/postsubmission_20260930'
FOLDER = REMOTE_PHASE + '/graph_relation_quality_recent_primary_search_20261008_v1'
RAW_NAMES = {
    'graph_ensemble_attention.atom', 'graph_ensemble_shared.atom',
    'graph_ensemble_diverse.atom', 'title_graph_ensemble.atom',
    'shared_graph_ensemble.atom', 'learned_graph_diversity.atom',
    'TWO_PAPER_METADATA.atom', '2503_14240v2.html', '2607_28304v1.html',
    'ph_crossref.json', 'pmlr_index.html', 'icml_2026_papers.html',
    'icml_2026_consensus_poster.html', 'DISCOVERY.json',
    'DISCOVERY_TARGETED.json',
}
SSH = ['ssh', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-p', '2222', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes',
       'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']

REMOTE = r'''
import base64,hashlib,json,os,stat,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
folder=phase/'graph_relation_quality_recent_primary_search_20261008_v1'
os.chdir(repo)
host=subprocess.check_output(['hostname'],text=True).strip()
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert host=='anogena-2-0'
assert gpus==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert repo.resolve()==repo and subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-toplevel'],text=True).strip()==str(repo)
assert phase.is_dir() and phase.resolve()==phase
p=json.load(sys.stdin)
assert p['folder']==str(folder) and set(p['files'])==EXPECTED_NAMES
assert p['mode'] in ('transfer','independent_verification')
if p['mode']=='transfer':
    os.umask(0o077)
    folder.mkdir(mode=0o700,exist_ok=True)
assert folder.is_dir() and folder.resolve()==folder
observed=[]
for name,row in p['files'].items():
    assert '/' not in name and '\\' not in name and name not in ('.','..')
    path=folder/name
    if p['mode']=='transfer':
        data=base64.b64decode(row['base64'],validate=True)
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        if not path.exists():
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as f:
                f.write(data);f.flush();os.fsync(f.fileno())
    st=path.lstat()
    assert stat.S_ISREG(st.st_mode) and path.resolve()==path
    body=path.read_bytes()
    actual={'name':name,'server_path':str(path),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
    assert actual['bytes']==row['bytes'] and actual['sha256']==row['sha256']
    observed.append(actual)
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'mode':p['mode'],'host':host,
 'visible_gpu_uuids':gpus,'directory':str(folder),'identity_assertions_passed_before_writes':True,
 'files':observed,'all_hashes_verified':True}))
'''


def bound(path):
    body=path.read_bytes()
    return dict(path=str(path.relative_to(PHASE)),bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest())


def remote(payload):
    script=REMOTE.replace('EXPECTED_NAMES',repr(RAW_NAMES))
    result=subprocess.run(SSH+['python3 -B -c '+shlex.quote(script)],
                          input=json.dumps(payload),capture_output=True,text=True,timeout=60)
    if result.returncode:
        raise RuntimeError('Custody operation failed: '+result.stderr)
    return json.loads(result.stdout)


def main():
    preflight=json.loads((HERE/'PREWRITE_ROUTE_VERIFICATION.json').read_text())
    assert preflight['exit_code']==0 and preflight['verification']['identity_assertions_passed']
    original_manifest=bound(HERE/'ARTIFACT_MANIFEST.json')
    rows={name:bound(HERE/name) for name in sorted(RAW_NAMES)}
    assert all(stat.S_ISREG((HERE/name).lstat().st_mode) for name in RAW_NAMES)
    saved_queries=[]
    for name in ('DISCOVERY.json','DISCOVERY_TARGETED.json'):
        record=json.loads((HERE/name).read_text())
        for q in record['queries']:
            saved_queries.append({k:q[k] for k in ('key','query','url','status')})
            saved_queries[-1]['returned_entries']=len(q['results'])
    (HERE/'DISCOVERY_SUMMARY.json').write_text(json.dumps(dict(
        queries=saved_queries,discovery_is_not_paper_read_credit=True,
        full_responses_and_abstracts_in_server_custody=True,
        selected_primary_identities=['arxiv:2503.14240v2','arxiv:2607.28304v1']),indent=2)+'\n')
    selection=dict(files=rows,original_artifact_manifest=original_manifest,
                   raw_bodies_only=True,compact_reports_scopes_and_excerpts_retained=True)
    (HERE/'LOCAL_RAW_OFFLOAD_SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
    payload=dict(mode='transfer',folder=FOLDER,files={
        name:dict(bytes=row['bytes'],sha256=row['sha256'],
                  base64=base64.b64encode((HERE/name).read_bytes()).decode('ascii'))
        for name,row in rows.items()})
    transfer=remote(payload)
    (HERE/'PRIMARY_OFFLOAD_TRANSFER_RECEIPT.json').write_text(json.dumps(transfer,indent=2)+'\n')
    verification=remote(dict(mode='independent_verification',folder=FOLDER,
                              files={name:dict(bytes=row['bytes'],sha256=row['sha256'])
                                     for name,row in rows.items()}))
    assert verification['all_hashes_verified'] and len(verification['files'])==len(RAW_NAMES)
    (HERE/'INDEPENDENT_SERVER_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    before_remove=datetime.now(timezone.utc).isoformat()
    for name,row in rows.items():
        assert bound(HERE/name)==row
    for name in rows:
        (HERE/name).unlink()
    files=[]
    remote_rows={r['name']:r for r in verification['files']}
    for name,row in rows.items():
        files.append(dict(name=name,original_local_path=row['path'],bytes=row['bytes'],
                          sha256=row['sha256'],server_path=remote_rows[name]['server_path'],
                          server_bytes=remote_rows[name]['bytes'],server_sha256=remote_rows[name]['sha256'],
                          local_body_removed_after_independent_server_verification=True,
                          local_body_removal_UTC=before_remove))
    custody=dict(schema='exact-literature-body-server-custody-v1',UTC=before_remove,
                 host=verification['host'],visible_gpu_uuids=verification['visible_gpu_uuids'],
                 authorized_root=REMOTE_PHASE,directory=FOLDER,
                 ssh_transport=preflight['transport'],
                 identity_assertions_passed_before_writes=True,
                 transfer_receipt=bound(HERE/'PRIMARY_OFFLOAD_TRANSFER_RECEIPT.json'),
                 independent_verification_receipt=bound(HERE/'INDEPENDENT_SERVER_VERIFICATION.json'),
                 original_artifact_manifest=original_manifest,files=files,
                 complete=True,raw_processed_bodies_only_offloaded=True,
                 local_raw_bodies_present=False,scientific_runs=0,
                 pending_outcomes_accessed=False,canonical_pointer_changes=0,
                 credential_values_recorded=False)
    (HERE/'SERVER_CUSTODY.json').write_text(json.dumps(custody,indent=2)+'\n')
    pointers=dict(schema='offloaded-literature-body-pointers-v1',UTC=before_remove,
                  status='EXACT_RAW_BODIES_IN_VERIFIED_ALLOCATION_PROJECT_CUSTODY',
                  local_path_semantics='Immutable acquisition locators now refer to exact retained bodies at server_path.',
                  original_scope_retrieval_and_artifact_metadata_preserved=True,
                  original_artifact_manifest=original_manifest,
                  server_custody=bound(HERE/'SERVER_CUSTODY.json'),
                  bodies=files,method_excerpts_retained=[bound(p) for p in sorted(HERE.glob('*METHOD*.txt'))],
                  original_reports_conclusions_and_read_scopes_retained=True,
                  new_primary_reads_in_offload=0,scientific_runs=0)
    (HERE/'RAW_PRIMARY_POINTERS.json').write_text(json.dumps(pointers,indent=2)+'\n')
    assert bound(HERE/'ARTIFACT_MANIFEST.json')==original_manifest
    assert all(not (HERE/name).exists() for name in RAW_NAMES)
    (HERE/'LOCAL_AFTER_OFFLOAD_MANIFEST.json').write_text(json.dumps(dict(
        original_acquisition_manifest_preserved=original_manifest,
        remote_raw_body_custody=bound(HERE/'SERVER_CUSTODY.json'),
        retained_local_files=[bound(p) for p in sorted(HERE.iterdir())
                              if p.is_file() and p.name!='LOCAL_AFTER_OFFLOAD_MANIFEST.json']),indent=2)+'\n')
    print(json.dumps(dict(complete=True,raw_bodies_offloaded=len(rows),
        removed_local_bytes=sum(r['bytes'] for r in rows.values()),
        custody=bound(HERE/'SERVER_CUSTODY.json'),pointers=bound(HERE/'RAW_PRIMARY_POINTERS.json')),
        indent=2))


if __name__=='__main__':
    main()
