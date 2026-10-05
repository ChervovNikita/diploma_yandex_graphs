"""One native CUDA resource check: full input shapes, zero fits/optimizer steps."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
ROOT=PHASE/'citeseer_ncn_native_runtime_qualification_20261005_v1'
SOURCE=PHASE/'citeseer_heart_ncn_trainval_runner_source_20261005_v1'
DATA=PHASE/'citeseer_heart_official_acquisition_server_20261005_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd()==REPO and socket.gethostname()=='anogena-2-0'
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (ROOT/'RESULT.json').exists() and not (ROOT/'FAILURE.json').exists()
    manifest=SOURCE/'SOURCE_MANIFEST.json'
    for r in json.loads(manifest.read_text())['files']:
        assert sha(SOURCE/r['path'])==r['sha256']
    qualification=PHASE/'citeseer_feature_qualification_20261005_v2/RESULT.json'
    assert json.loads(qualification.read_text())['status']=='PASS'
    started=time.monotonic()
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),fits=0,optimizer_updates=0,
                 TEST_access=False,ranking_metrics_computed=False,
                 source_manifest_sha256=sha(manifest),source_sha256=sha(Path(__file__)),
                 data_qualification_sha256=sha(qualification))
    try:
        import numpy as np
        import torch
        import torch_geometric
        import torch_sparse
        import torch_scatter
        from torch_sparse import SparseTensor
        from torch_geometric.utils import to_undirected,negative_sampling
        sys.path.insert(0,str(SOURCE))
        import run
        from heads import make_predictor
        assert Path(run.__file__).resolve()==SOURCE/'run.py'
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        available=DATA/'AVAILABLE_MANIFEST.json'
        job=dict(available_manifest_relative=str(available.relative_to(PHASE)),
                 available_manifest_sha256=sha(available),qualified_feature_shape=[3327,3703])
        x,train,valid,pool,identities=run.load_available(job)
        run.seed_native(0)
        native,PermIterator=run.load_native()
        encoder=native.GCN(x.size(1),256,256,1,.3,True,False,-1,'puregcn',True,0.,
                           xdropout=.4,taildropout=0.,noinputlin=False).cuda()
        predictor=make_predictor(native,arm='native_single',members=1).cuda()
        x,train,valid,pool=[v.cuda() for v in (x,train,valid,pool)]
        edge=to_undirected(train.t())
        adj=SparseTensor.from_edge_index(edge,sparse_sizes=(len(x),len(x))).to_symmetric().coalesce()
        torch.cuda.reset_peak_memory_stats()
        negatives=negative_sampling(edge,len(x))
        perm=next(iter(PermIterator(x.device,len(train),1024)))
        keep=torch.ones(len(train),device=x.device,dtype=torch.bool);keep[perm]=False
        temporary=SparseTensor.from_edge_index(train[keep].t(),sparse_sizes=(len(x),len(x))).to_symmetric()
        encoder.train();predictor.train()
        h=encoder(x,temporary)
        pos=predictor(h,temporary,train[perm].t())
        neg=predictor(h,temporary,negatives[:,perm])
        assert pos.shape==neg.shape==(1024,1)
        loss=-torch.nn.functional.logsigmoid(pos).mean()-torch.nn.functional.logsigmoid(-neg).mean()
        assert torch.isfinite(loss);loss.backward()
        gradients=[p.grad for p in [*encoder.parameters(),*predictor.parameters()] if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        gradient_tensor_count=len(gradients)
        for p in [*encoder.parameters(),*predictor.parameters()]:p.grad=None
        del pos,neg,loss,h,temporary,gradients
        encoder.eval();predictor.eval()
        with torch.no_grad():
            h=encoder(x,adj)
            # Full released VALID graph/pool shape; no labels-to-scores metric.
            positive=predictor(h,adj,valid.t())
            negative=predictor(h,adj,pool.permute(2,0,1).reshape(2,-1))
            assert positive.shape==(227,1) and negative.shape==(113500,1)
            assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
        torch.cuda.synchronize()
        receipt.update(status='PASS',runtime_versions=dict(torch=str(torch.__version__),numpy=np.__version__,
                       torch_geometric=torch_geometric.__version__,torch_sparse=torch_sparse.__version__,
                       torch_scatter=torch_scatter.__version__,CUDA=torch.version.cuda),
                       available_manifest_sha256=sha(available),input_identities=identities,
                       masked_TRAIN_queries=1024,VALID_positive_queries=227,VALID_negative_queries=113500,
                       source_native_single_operators_qualified=True,predictive_competence=False,
                       gradient_parameter_tensors_checked=gradient_tensor_count,
                       peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
                       inclusive_seconds=time.monotonic()-started)
        filename='RESULT.json'
    except Exception as error:
        receipt.update(status='FAIL',error=type(error).__name__+': '+str(error),
                       inclusive_seconds=time.monotonic()-started,partial_files_preserved=True)
        filename='FAILURE.json'
        with (ROOT/filename).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
        raise
    with (ROOT/filename).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
