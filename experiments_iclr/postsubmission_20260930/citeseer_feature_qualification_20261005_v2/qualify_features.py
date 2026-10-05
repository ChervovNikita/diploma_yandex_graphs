"""Authenticate public Citeseer inputs on CPU; never open withheld test inputs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import inspect
import json
import os
import socket
import subprocess
import time
import urllib.request

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
ROOT = PHASE / 'citeseer_feature_qualification_20261005_v2'
ACQUISITION = PHASE / 'citeseer_heart_official_acquisition_server_20261005_v1'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ROOT / name).open('x') as handle:
        json.dump(value, handle, indent=2);handle.write('\n')


def main():
    assert Path.cwd()==REPO and socket.gethostname()=='anogena-2-0'
    assert Path(__file__).resolve().parent==ROOT
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                          capture_output=True,text=True,check=True).stdout.splitlines()==[UUID]
    assert not (ROOT / 'RESULT.json').exists() and not (ROOT / 'FAILURE.json').exists()
    started=time.monotonic()
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),
                GPU_UUID=UUID,CUDA_VISIBLE_DEVICES='',fits=0,optimizer_updates=0,
                prediction_outcomes_read=False,TEST_inputs_read=False,
                source_sha256=sha(Path(__file__)),input_acquisition=str(ACQUISITION))
    try:
        import numpy as np
        import torch
        import torch_geometric
        from torch_geometric.datasets import Planetoid
        manifest_path=ACQUISITION / 'AVAILABLE_MANIFEST.json'
        manifest=json.loads(manifest_path.read_text())
        assert manifest['TEST_available_to_loader'] is False
        inputs={}
        for name,row in manifest['files'].items():
            path=ACQUISITION / row['relative_path']
            assert path.resolve().is_relative_to(ACQUISITION / 'available')
            assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            inputs[name]=path
        assert set(inputs)=={'train_pos.txt','valid_pos.txt','heart_valid_samples.npy','gnn_feature'}
        feature_payload=torch.load(inputs['gnn_feature'],map_location='cpu',weights_only=True)
        assert isinstance(feature_payload,dict) and 'entity_embedding' in feature_payload
        supplied=feature_payload['entity_embedding']
        assert isinstance(supplied,torch.Tensor) and supplied.layout==torch.strided
        assert supplied.ndim==2 and supplied.is_floating_point() and torch.isfinite(supplied).all()
        result['supplied_feature']=dict(shape=list(supplied.shape),dtype=str(supplied.dtype),
                                       payload_keys=sorted(feature_payload),sha256=sha(inputs['gnn_feature']))
        # Bind the independent raw-feature donor to an immutable public commit.
        retained_commit=PHASE / 'citeseer_feature_qualification_20261005_v1/PLANETOID_COMMIT.json'
        commit=json.loads(retained_commit.read_text())
        result['retained_commit_metadata']=dict(path=str(retained_commit),sha256=sha(retained_commit))
        pin=commit['sha']
        assert len(pin)==40 and all(c in '0123456789abcdef' for c in pin)
        save('PLANETOID_COMMIT.json',commit)
        Planetoid.url='https://raw.githubusercontent.com/kimiyoung/planetoid/'+pin+'/data'
        dataset=Planetoid(root=str(ROOT / 'raw_reference'),name='Citeseer',transform=None,
                          pre_transform=None)
        reference=dataset[0].x
        equal=supplied.shape==reference.shape and supplied.dtype==reference.dtype and torch.equal(supplied,reference)
        result['raw_reference']=dict(repository='kimiyoung/planetoid',commit=pin,
                                    URL=Planetoid.url,torch_version=torch.__version__,
                                    torch_geometric_version=torch_geometric.__version__,
                                    Planetoid_loader_sha256=sha(Path(inspect.getfile(Planetoid))),
                                    shape=list(reference.shape),dtype=str(reference.dtype),
                                    raw_files=[dict(name=Path(f).name,bytes=Path(f).stat().st_size,sha256=sha(Path(f))) for f in dataset.raw_paths],
                                    all_rows_and_columns_bitwise_equal=equal)
        assert equal, 'Released feature does not exactly equal independent raw Planetoid features.'
        positives={}
        for split in ('train','valid'):
            rows=np.loadtxt(inputs[split+'_pos.txt'],dtype=np.int64,delimiter='\t',ndmin=2)
            assert rows.shape[1]==2 and rows.min()>=0 and rows.max()<supplied.shape[0]
            positives[split]=rows[rows[:,0]!=rows[:,1]]
            result[split+'_positives']=dict(raw_rows=len(rows),self_loops_removed=int((rows[:,0]==rows[:,1]).sum()),native_rows=len(positives[split]))
        pool=np.load(inputs['heart_valid_samples.npy'],allow_pickle=False)
        valid=positives['valid']
        assert pool.shape==(len(valid),500,2) and pool.dtype==np.dtype('int64')
        assert pool.min()>=0 and pool.max()<supplied.shape[0]
        first_source=np.all(pool[:,:250,0]==valid[:,0,None])
        second_target=np.all(pool[:,250:,1]==valid[:,1,None])
        result['VALID_pool']=dict(shape=list(pool.shape),dtype=str(pool.dtype),
                                 min_ID=int(pool.min()),max_ID=int(pool.max()),
                                 first250_match_positive_source=bool(first_source),
                                 last250_match_positive_target=bool(second_target),
                                 duplicates_and_native_collision_semantics_unmodified=True)
        assert first_source and second_target, 'Pool rows do not match recorded endpoint ordering.'
        result.update(status='PASS',features_raw_and_row_IDs_verified=True,
                      TRAIN_VALID_input_qualification=True,predictive_competence=False,
                      elapsed_seconds=time.monotonic()-started)
        save('RESULT.json',result)
    except Exception as error:
        result.update(status='FAIL',error=type(error).__name__+': '+str(error),
                      elapsed_seconds=time.monotonic()-started,files_preserved=True)
        save('FAILURE.json',result)
        raise
    print(json.dumps(result))


if __name__=='__main__':
    main()
