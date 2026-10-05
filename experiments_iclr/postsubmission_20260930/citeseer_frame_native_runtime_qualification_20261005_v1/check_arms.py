"""Qualify exact new NCN arms on full TRAIN/VALID shapes; no fits or steps."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import socket
import subprocess
import sys
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
ROOT=PHASE/'citeseer_frame_native_runtime_qualification_20261005_v1'
SOURCE=PHASE/'citeseer_heart_ncn_trainval_runner_source_20261005_v1'
COHORT=PHASE/'citeseer_endpoint_frame_paired_development_20261005_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd()==REPO and socket.gethostname()=='anogena-2-0'
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (ROOT/'RESULT.json').exists() and not (ROOT/'FAILURE.json').exists()
    manifest=SOURCE/'SOURCE_MANIFEST.json'
    for row in json.loads(manifest.read_text())['files']:assert sha(SOURCE/row['path'])==row['sha256']
    started=time.monotonic()
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),fits=0,optimizer_updates=0,
                 TEST_access=False,ranking_metrics_computed=False,
                 source_manifest_sha256=sha(manifest),source_sha256=sha(Path(__file__)),
                 plan_sha256=sha(COHORT/'PLAN.json'),checked_arms=[])
    try:
        import torch
        from torch_sparse import SparseTensor
        from torch_geometric.utils import to_undirected,negative_sampling
        sys.path.insert(0,str(SOURCE))
        import run
        from heads import make_predictor
        assert Path(run.__file__).resolve()==SOURCE/'run.py'
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        job=json.loads((COHORT/'b0_native_single_seed0.job.json').read_text())
        x,train,valid,pool,_=run.load_available(job)
        native,PermIterator=run.load_native()
        run.seed_native(0)
        encoder=native.GCN(x.size(1),256,256,1,.3,True,False,-1,'puregcn',True,0.,
                           xdropout=.4,taildropout=0.,noinputlin=False).cuda()
        x,train,valid,pool=[v.cuda() for v in (x,train,valid,pool)]
        edge=to_undirected(train.t())
        adj=SparseTensor.from_edge_index(edge,sparse_sizes=(len(x),len(x))).to_symmetric().coalesce()
        negatives=negative_sampling(edge,len(x))
        perm=next(iter(PermIterator(x.device,len(train),1024)))
        keep=torch.ones(len(train),device=x.device,dtype=torch.bool);keep[perm]=False
        temporary=SparseTensor.from_edge_index(train[keep].t(),sparse_sizes=(len(x),len(x))).to_symmetric()
        factor_seed=next(r['factor_seed'] for r in json.loads((COHORT/'PLAN.json').read_text())['physical_fits'] if r['arm']=='unframed_f4')
        cases=[(a,4,None) for a in ['unframed_f4','shared_frame_f4','private_frame_f4']]
        cases += [('same_four_frames_single',1,None)]
        cases += [('independent_frame_member',1,i) for i in range(4)]
        reference_bank=None;reference_single=None
        encoder.eval()
        with torch.no_grad():encoded=encoder(x,adj)
        for arm,members,axis in cases:
            torch.cuda.reset_peak_memory_stats();run.seed_native(0)
            predictor=make_predictor(native,arm=arm,members=members,axis_index=axis,
                                     factor_seed=factor_seed if members==4 else None).cuda()
            predictor.eval()
            with torch.no_grad():initial=predictor(encoded,adj,valid.t()).cpu()
            if arm=='unframed_f4':reference_bank=initial
            elif members==4:assert torch.equal(initial,reference_bank), 'Axis initialization must preserve bank eval logits.'
            else:
                run.seed_native(0)
                donor=make_predictor(native,arm='native_single',members=1).cuda().eval()
                with torch.no_grad():reference_single=donor(encoded,adj,valid.t()).cpu()
                assert torch.allclose(initial,reference_single,rtol=1.52587890625e-5,atol=1.52587890625e-5)
                del donor
            encoder.train();predictor.train()
            h=encoder(x,temporary)
            positive=predictor(h,temporary,train[perm].t())
            negative=predictor(h,temporary,negatives[:,perm])
            assert positive.shape==negative.shape==(1024,members)
            loss=-torch.nn.functional.logsigmoid(positive).mean()-torch.nn.functional.logsigmoid(-negative).mean()
            assert torch.isfinite(loss);loss.backward()
            grads=[p.grad for p in [*encoder.parameters(),*predictor.parameters()] if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            frame_nonzero=None
            if hasattr(predictor,'v'):
                assert predictor.v.grad is not None
                frame_nonzero=[bool(torch.count_nonzero(g)) for g in predictor.v.grad]
                assert all(frame_nonzero), 'Declared frame has no accessible gradient on the native batch.'
            for parameter in [*encoder.parameters(),*predictor.parameters()]:parameter.grad=None
            del positive,negative,loss,h,grads
            encoder.eval();predictor.eval()
            with torch.no_grad():
                h=encoder(x,adj)
                positive=predictor(h,adj,valid.t())
                negative=predictor(h,adj,pool.permute(2,0,1).reshape(2,-1))
                assert positive.shape==(227,members) and negative.shape==(113500,members)
                assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
            torch.cuda.synchronize()
            receipt['checked_arms'].append(dict(arm=arm,axis_index=axis,members=members,
                    native_initial_eval_parity=True,frame_gradients_nonzero=frame_nonzero,
                    full_VALID_negative_queries=113500,
                    peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved()))
            del positive,negative,h,predictor,initial
        receipt.update(status='PASS',inclusive_seconds=time.monotonic()-started)
        filename='RESULT.json'
    except Exception as error:
        receipt.update(status='FAIL',error=type(error).__name__+': '+str(error),inclusive_seconds=time.monotonic()-started)
        with (ROOT/'FAILURE.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
        raise
    with (ROOT/filename).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
