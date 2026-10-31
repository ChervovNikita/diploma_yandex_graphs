"""Synthetic CPU native typed mean/sparse preparation checks; no model."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation required before project access')
    root=Path(args.project_root).resolve();stage=Path(__file__).resolve().parent;out=root/args.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined CPU source check required')
    out.mkdir();os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(stage/'providers/dgl_cpu'));sys.path.insert(0,str(stage/'runtime'))
    from source_utils import bind_source,native_runtime,receipt
    try:
        source=bind_source(root);rt,engine=native_runtime(root,source,'cpu');t,np,dgl,S,native=(rt[k] for k in ('torch','numpy','dgl','SparseTensor','native'))
        t.set_num_threads(2)
        if dgl.__version__!='1.1.3' or not Path(dgl.__file__).resolve().is_relative_to(stage/'providers/dgl_cpu'):
            raise ValueError('Exact isolated CPU DGL provider required')
        symbols=('M','D','A','K');counts={'M':4,'D':2,'A':3,'K':3};removal={None:set(),'actor':{'MA','AM'},'director':{'MD','DM'},'keyword':{'MK','KM'}}
        pairs={'MD':[(0,0),(1,1),(2,0),(3,1)],'MA':[(0,0),(0,1),(1,1),(2,2),(3,0)],'MK':[(0,0),(1,1),(2,1),(3,2)]}
        pairs.update({k[::-1]:[(b,a) for a,b in v] for k,v in list(pairs.items())})
        generator=t.Generator().manual_seed(72901);features={s:t.randn(n,3,generator=generator) for s,n in counts.items()};features['K']=t.eye(3)
        checks=[]
        for family in (None,'actor','director','keyword'):
            adjs,edges={},{}
            for key,values in pairs.items():
                values=[] if key in removal[family] else values;r=t.LongTensor([a for a,b in values]);c=t.LongTensor([b for a,b in values])
                adj=S(row=r,col=c,sparse_sizes=(counts[key[0]],counts[key[1]]));adj.storage._value=None;adj.storage._value=t.ones(adj.nnz())/adj.sum(dim=-1)[adj.storage.row()]
                adjs[key]=adj;edges[(key[1],key[1]+'-'+key[0],key[0])]=(c.numpy(),r.numpy())
            graph=dgl.heterograph(edges,num_nodes_dict=counts)
            for s,v in features.items():graph.nodes[s].data[s]=v.clone()
            with contextlib.redirect_stdout(io.StringIO()):graph=native.hg_propagate_feat_dgl(graph,'M',4,5,[],echo=False)
            actual=dict(graph.nodes['M'].data);expected={s:{s:v.clone()} for s,v in features.items()}
            # Independent dense row means compose the same retained typed paths.
            for hop in range(1,5):
                for key,adj in adjs.items():
                    for path,v in list(expected[key[1]].items()):
                        if len(path)==hop and not (hop==4 and key[0]!='M'):expected[key[0]][key[0]+path]=adj.to_dense()@v
                for s in symbols:
                    if s!='M':expected[s]={k:v for k,v in expected[s].items() if len(k)>hop}
            if sorted(actual)!=rt['protocol']['feature_paths'] or set(actual)!=set(expected['M']):raise ValueError('Native full channel namespace changed')
            largest=0.
            for key,value in actual.items():
                ref=expected['M'][key];largest=max(largest,float((value-ref).abs().max()))
                if not t.allclose(value,ref,rtol=3e-6,atol=3e-6):raise ValueError('Native typed mean orientation/multi-hop composition differs')
            with contextlib.redirect_stdout(io.StringIO()):products=native.hg_propagate_sparse_pyg(adjs,'M',4,5,[],prop_feats=False,echo=False,prop_device='cpu')
            if sorted(products)!=rt['protocol']['label_paths']:raise ValueError('Native return-product namespace changed')
            train=t.zeros(4,5);train[0]=t.tensor([1.,0.,1.,0.,1.]);train[2]=t.tensor([0.,1.,0.,1.,0.])
            for key,value in products.items():
                dense=value.to_dense().clone();dense.fill_diagonal_(0)
                native_labels=rt['remove_diag'](value)@train
                if not t.allclose(native_labels,dense@train,rtol=3e-6,atol=3e-6):raise ValueError('Diagonal removal or TRAIN-only label multiplication differs')
            if not t.equal(actual['M'],features['M']):raise ValueError('Movie-own features changed under raw-family removal')
            removed=None if family is None else {'actor':'A','director':'D','keyword':'K'}[family]
            if removed and any(t.count_nonzero(v) for k,v in actual.items() if removed in k):raise ValueError('Deleted paired raw support left dependent message channels')
            if any(graph.num_nodes(s)!=n for s,n in counts.items()):raise ValueError('Isolated typed nodes lost')
            checks.append({'family':family,'feature_channels':25,'label_products':12,'max_abs_dense_reference_error':largest,'mean_sparse_diagonal_removed_TRAIN_only_isolated_node_checks':'pass'})
        if t.cuda.is_initialized():raise ValueError('Synthetic CPU check initialized CUDA')
        provider_files=[receipt(root,p) for p in sorted((stage/'providers/dgl_cpu').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
        manifest={'schema':'isolated-project-CPU-DGL-provider-v1','provider':'dgl','version':str(dgl.__version__),'provider_root':str((stage/'providers/dgl_cpu').relative_to(root)),
            'files':provider_files,'cuda_dependency':'CPU DGL only; source graph preprocessing is CPU. Native NN learning remains Torch CUDA after admission and GO.',
            'shared_venv_DGL_replaced':False,'later_77_sync':'Copy provider target with exact bytes; prepend its root to sys.path/PYTHONPATH for source helpers; keep Python3.11/torch2.1.2+cu118/numpy1.26.4/torch-sparse0.6.18+pt21cu118 compatible and qualify source import/CPU operators there.',
            'source_function_receipts':source['modules'],'checks':checks,'fits':0,'optimizer_updates':0,'predictive_scores':0,'CUDA':False,'GO':False}
        (out/'PROVIDER_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        result={'status':'passed','scope':'Synthetic typed source preprocessing only; no model constructor, gradient fit, CUDA or scientific score','checks':checks,
            'provider_manifest':receipt(root,out/'PROVIDER_MANIFEST.json'),'torch':str(t.__version__),'numpy':str(np.__version__),'DGL':str(dgl.__version__),
            'fits':0,'optimizer_updates':0,'predictive_scores':0,'CUDA':False,'GO':False}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'passed','typed_graph_cases':4,'native_feature_channel_cases':100,'native_label_product_cases':48,'provider_manifest':result['provider_manifest'],'fits':0,'CUDA':False,'GO':False}))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'fits':0,'CUDA':False,'GO':False},indent=2)+'\n');raise


if __name__=='__main__':main()
