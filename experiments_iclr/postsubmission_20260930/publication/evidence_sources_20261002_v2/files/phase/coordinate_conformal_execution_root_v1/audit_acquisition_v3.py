"""Independently audit label-free role arrays and retained supervision receipts."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

PHASE=Path(__file__).resolve().parents[1]
ROOT=PHASE/'coordinate_conformal_execution_root_v1'
REMOTE='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def local(record):
    if not record['path'].startswith(REMOTE):
        raise ValueError('Descriptor outside authorized phase')
    path=PHASE/record['path'][len(REMOTE):]
    if not path.resolve().is_relative_to(PHASE) or sha(path)!=record['sha256']:
        raise ValueError('Descriptor mismatch')
    return path

def npy(data):
    stream=io.BytesIO(data)
    if stream.read(6)!=b'\x93NUMPY':
        raise ValueError('Unexpected NPY magic')
    major,minor=stream.read(2)
    size=struct.unpack('<H' if major==1 else '<I',stream.read(2 if major==1 else 4))[0]
    header=ast.literal_eval(stream.read(size).decode('latin1'))
    body=stream.read()
    return header,body

def ids(path):
    header,body=npy(path.read_bytes())
    if header['descr']!='<i8' or len(header['shape'])!=1:
        raise ValueError('Role dtype or shape differs')
    count=header['shape'][0]
    if len(body)!=count*8:
        raise ValueError('Role payload length differs')
    values=struct.unpack('<'+'q'*count,body)
    if tuple(sorted(set(values)))!=values:
        raise ValueError('Role IDs not sorted/unique')
    return set(values)

def main():
    output=ROOT/'ACQUISITION_ASSESSMENT_v3.json'
    if output.exists():
        raise ValueError('Audit identity already used')
    outer=read(ROOT/'acquisition_bound_run02/TERMINAL.json')
    assert outer['complete'] and outer['within_whole_cap'] and not outer['timed_out']
    assert outer['root_request_unchanged'] and outer['child_exit_code']==0
    assert sha(ROOT/'acquisition_bound_run02/START.json')==outer['START_sha256']
    for name,item in outer['inner_evidence'].items():
        assert sha(ROOT/'acquisition_supervisor_run02'/name)==item['sha256']
    authorization=read(ROOT/'acquisition_supervisor_run02/authorization.json')
    assert authorization['visible_gpu_count']==1
    assert authorization['gpu_uuid']=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    request_path=ROOT/'ACQUISITION_REQUEST_v2.json';request=read(request_path)
    terminal=read(ROOT/'acquisition_run02/TERMINAL.json')
    assert terminal['complete'] and terminal['training_steps']==0 and terminal['heldout_scores']==0
    assert terminal['request_sha256']==sha(request_path)
    local(terminal['fit_environment'])
    mask_record=read(ROOT/'acquisition_run02/Squirrel/GRAPH_INPUT.json')['published_masks']
    with zipfile.ZipFile(local(mask_record)) as archive:
        masks={}
        for name in ('train','validation','pool'):
            header,body=npy(archive.read(name+'.npy'))
            assert header['descr']=='|b1' and header['shape'][1]==2223 and header['shape'][0]>=3
            assert len(body)==header['shape'][0]*2223
            nrows,ncols=header['shape']
            masks[name]=[{i for i in range(ncols)
                          if body[split+i*nrows if header['fortran_order'] else split*ncols+i]}
                         for split in range(3)]
    cases=[]
    for descriptor in terminal['graphs']:
        manifest_path=local(descriptor);manifest=read(manifest_path)
        graph_path=local(manifest['graph']);graph=read(graph_path)
        assert manifest['fits_or_scores_performed'] is False
        assert manifest['final_pool_labels_supplied_to_fit'] is False
        assert len(manifest['cells'])==3
        for split,cell in enumerate(manifest['cells']):
            freeze_path=local(cell['role_freeze']);freeze=read(freeze_path)
            assert freeze['labels_read'] is False
            assert freeze['preparation_driver_sha256']==request['driver']['sha256']
            assert cell['roles_frozen_before_label_extraction'] is True
            assert cell['seed']==(17,29,43)[split] and cell['source_split_index']==split
            for payload in freeze['payload']:
                assert sha(freeze_path.parent/payload['path'])==payload['sha256']
            roles={name:ids(freeze_path.parent/(name+'_nodes.npy')) for name in ('train','validation','A','B','D','pool')}
            counts={name:len(value) for name,value in roles.items()}
            assert counts==freeze['source_counts']==cell['source_counts']
            n=graph['num_nodes']
            assert sum(counts.values())==n and set.union(*roles.values())==set(range(n))
            assert all(roles[a].isdisjoint(roles[b]) for a in roles for b in roles if a!=b)
            assert counts['train']==n*20//100 and counts['pool']==n-n*50//100
            if graph['graph']=='Squirrel':
                assert counts==dict(train=444,validation=222,A=111,B=222,D=112,pool=1112)
                assert masks['pool'][split]<=roles['pool']
                assert roles['train']<=masks['train'][split]
                source_reservoir=set.union(*(roles[name] for name in ('validation','A','B','D')))
                assert source_reservoir<=masks['validation'][split]
                assert len(masks['validation'][split]&roles['pool'])==(51,33,59)[split]
            cases.append(dict(graph=graph['graph'],seed=cell['seed'],source_split_index=split,
                              counts=counts,role_freeze_sha256=sha(freeze_path),partition_verified=True))
    value=dict(schema='gnnm-label-free-acquisition-assessment-v3',UTC=datetime.now(timezone.utc).isoformat(),
               request_sha256=sha(request_path),outer_terminal_sha256=sha(ROOT/'acquisition_bound_run02/TERMINAL.json'),
               complete=True,cases=cases,source_and_final_labels_read_by_auditor=False,
               training_or_scores=0,whole_supervised_seconds=outer['whole_supervised_seconds'],
               previous_failure_preserved=True,
               scope='Independent stdlib role/payload/supervision audit; does not qualify models or certify heldout utility.')
    with output.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps({'complete':True,'cases':len(cases),'assessment_sha256':sha(output)}))

if __name__=='__main__':
    main()
