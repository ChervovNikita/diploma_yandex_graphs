"""UNEXECUTED future custodian source. Do not run under source-only admission.

Stage1 freezes official RL role IDs using public features/masks, without opening
any targets member. Stage2 produces compact train/validation packs only. Test
values are never converted to numbers or included in output. Runtime isolation
must exclude the raw archive and future test packs from scientific workers.
Authorization JSON is a provenance gate, not a security/signature mechanism.
"""
from __future__ import annotations
import csv,datetime,hashlib,io,json,math,os,zipfile
from pathlib import Path

ARCHIVE_SHA256='90c38ab363dd57ce67ec62e1501532a5189403119c9cf4791a91f127321a3ad6'
ARCHIVE_BYTES=3361114
PUBLIC_MEMBERS=('info.yaml','features.csv','edgelist.csv','split_masks_RL.csv')

def _hash(data):return hashlib.sha256(data).hexdigest()
def _write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n');os.chmod(path,0o600)
def _authorize(path,phase,policy_path):
    x=json.loads(Path(path).read_text())
    if x.get('phase')!=phase or x.get('source_only') is not False:
        raise ValueError('A separate trusted coordinator phase admission is required')
    if x.get('role_policy_sha256')!=_hash(Path(policy_path).read_bytes()):
        raise ValueError('Frozen role policy differs')
    return x

def _archive(path):
    p=Path(path)
    # This I/O is future custodian acquisition, never preparation-mode execution.
    b=p.read_bytes()
    if len(b)!=ARCHIVE_BYTES or _hash(b)!=ARCHIVE_SHA256:
        raise ValueError('Archive differs from acquired receipt')
    z=zipfile.ZipFile(io.BytesIO(b))
    if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP member names')
    return z

def _csv(blob):
    return list(csv.reader(io.StringIO(blob.decode('utf-8-sig'),newline='')))
def _bool(text):
    x=text.strip().lower()
    if x not in ('0','1','false','true'):raise ValueError('Nonboolean official role mask')
    return x in ('1','true')

def freeze_public_roles(archive,output_dir,policy_path,authorization_path):
    _authorize(authorization_path,'public_structure_role_freeze',policy_path)
    policy=json.loads(Path(policy_path).read_text())
    if policy['split']!='RL' or policy['test_label_state']!='CLOSED':raise ValueError('Role policy differs')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False);os.chmod(out,0o700)
    with _archive(archive) as z:
        public={name:z.read('tolokers-2/'+name) for name in PUBLIC_MEMBERS}
        # targets.csv is intentionally never opened in stage1.
    features=_csv(public['features.csv']);masks=_csv(public['split_masks_RL.csv'])
    ids=[r[0] for r in features[1:]]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Feature ids are empty or duplicated')
    if [r[0] for r in masks[1:]]!=ids:raise ValueError('Feature/official mask row order differs')
    cols={name:masks[0].index(name) for name in ('train','val','test')}
    role_ids={name:[] for name in cols};unassigned=[]
    for node_id,row in zip(ids,masks[1:]):
        bits={name:_bool(row[index]) for name,index in cols.items()}
        if sum(bits.values())>1:raise ValueError('Official roles overlap')
        if not any(bits.values()):unassigned.append(node_id)
        else:role_ids[next(name for name,b in bits.items() if b)].append(node_id)
    raw=out/'public_raw';raw.mkdir();os.chmod(raw,0o700)
    descriptors={}
    for name,blob in public.items():
        (raw/name).write_bytes(blob)
        descriptors[name]={'bytes':len(blob),'sha256':_hash(blob)}
    roles={'schema':'tolokers2-public-role-freeze-v1','dataset':'tolokers-2','split':'RL',
           'source_archive_sha256':ARCHIVE_SHA256,'role_policy_sha256':_hash(Path(policy_path).read_bytes()),
           'node_ids':ids,'official_role_ids':role_ids,'unassigned_node_ids':unassigned,'public_members':descriptors,
           'targets_member_opened':False,'test_labels':'CLOSED',
           'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    _write(out/'ROLE_FREEZE.json',roles)
    return {'role_freeze_sha256':_hash((out/'ROLE_FREEZE.json').read_bytes()),'labels_opened':False}

def derive_compact_trainval(archive,role_freeze_path,output_dir,policy_path,authorization_path):
    admission=_authorize(authorization_path,'compact_train_validation_label_acquisition',policy_path)
    # Fail before target acquisition or pack creation. A coordinator attestation
    # alone does not establish isolation; actual mount/access review is required.
    if admission.get('worker_mount_excludes_raw_archive_and_targets') is not True:
        raise ValueError('Custodian/worker file isolation must be admitted before target acquisition')
    rp=Path(role_freeze_path);roles=json.loads(rp.read_text())
    freeze_sha=_hash(rp.read_bytes())
    if admission.get('role_freeze_sha256')!=freeze_sha:raise ValueError('Roles were not frozen before labels')
    if roles.get('schema')!='tolokers2-public-role-freeze-v1' or roles.get('dataset')!='tolokers-2' or roles.get('split')!='RL':
        raise ValueError('Role freeze identity differs')
    if roles.get('source_archive_sha256')!=ARCHIVE_SHA256 or roles.get('role_policy_sha256')!=_hash(Path(policy_path).read_bytes()):
        raise ValueError('Role freeze archive/policy binding differs')
    if roles['targets_member_opened'] or roles['test_labels']!='CLOSED':raise ValueError('Invalid prior role freeze')
    train=set(roles['official_role_ids']['train']);val=set(roles['official_role_ids']['val'])
    packs={'train':[],'val':[]};expected_ids=iter(roles['node_ids'])
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False);os.chmod(out,0o700)
    with _archive(archive) as z,z.open('tolokers-2/targets.csv') as member:
        # Custodian alone parses the CSV syntax. Non-source values are skipped
        # before semantic conversion; no test class/missingness/statistics emitted.
        reader=csv.reader(io.TextIOWrapper(member,encoding='utf-8-sig',newline=''))
        header=next(reader)
        if len(header)!=2:raise ValueError('Expected one target column and node id')
        for row in reader:
            node_id=next(expected_ids,None)
            if node_id is None or len(row)!=2 or row[0]!=node_id:raise ValueError('Target/role axis mismatch')
            role='train' if node_id in train else 'val' if node_id in val else None
            if role is None:continue
            raw=row[1].strip()
            if raw.lower() in ('','nan','na','null','none'):continue
            y=float(raw)
            if not math.isfinite(y) or y not in (0.,1.):raise ValueError('Source binary target invalid')
            packs[role].append({'node_id':node_id,'target':int(y)})
        if next(expected_ids,None) is not None:raise ValueError('Target rows are missing')
    for role,records in packs.items():
        if not records or {r['target'] for r in records}!={0,1}:raise ValueError('Source role lacks both classes')
    receipts=[]
    for role,records in packs.items():
        payload={'schema':'compact-source-labelpack-v1','role':role,'dataset':'tolokers-2','split':'RL',
                 'role_freeze_sha256':freeze_sha,'source_archive_sha256':ARCHIVE_SHA256,'records':records}
        p=out/(role+'_labels.json');_write(p,payload)
        receipts.append({'role':role,'file':p.name,'sha256':_hash(p.read_bytes()),'nonmissing_count':len(records)})
    # Worker gets only these two packs, public_raw and feature/graph artifacts.
    _write(out/'LABELPACK_RECEIPT.json',{'source_roles':receipts,'role_freeze_sha256':freeze_sha,
        'test_values_semantically_decoded':False,'test_statistics_emitted':False,
        'worker_mount_excludes_raw_archive_and_targets':admission.get('worker_mount_excludes_raw_archive_and_targets') is True})
    return {'labelpack_receipt_sha256':_hash((out/'LABELPACK_RECEIPT.json').read_bytes()),'test_labels':'CLOSED'}
