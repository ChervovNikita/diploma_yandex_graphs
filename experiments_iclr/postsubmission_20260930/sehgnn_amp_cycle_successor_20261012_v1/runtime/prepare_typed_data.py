"""Independent full HGB IMDB source custody and CPU preprocessing; no model."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace

from source_utils import bind_source,bound,load,receipt,native_runtime,RAW_FILES

ROLE_RULE='RandomState(99701+1009*split).shuffle(sorted released development IDs); floor50percent TRAIN, floor20percent SELECT, floor10percent CAL, remaining development EVALUATE; no label.dat.test access'
FAMILIES=(None,'actor','director','keyword')
SYMBOLS=('M','D','A','K')
REMOVAL={'actor':{'MA','AM'},'director':{'MD','DM'},'keyword':{'MK','KM'},None:set()}


class Costs:
    def __init__(self,folder):self.folder=folder;self.rows=[]
    @contextmanager
    def measure(self,scope,gpu=False):
        if gpu:raise ValueError('Preprocessing never permits a GPU cost scope')
        started=time.monotonic();usage=resource.getrusage(resource.RUSAGE_SELF);row={'scope':scope}
        try:yield row;row['status']='complete'
        except BaseException as e:row.update(status='failed',error=str(e));raise
        finally:
            end=resource.getrusage(resource.RUSAGE_SELF)
            row.update(seconds=time.monotonic()-started,user_seconds=end.ru_utime-usage.ru_utime,system_seconds=end.ru_stime-usage.ru_stime,
                process_lifetime_host_RSS_highwater_bytes=end.ru_maxrss*1024)
            self.rows.append(row)
            with (self.folder/'COST_EVENTS.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')


def typed_static(rt,data,family,costs):
    """Same native DGL means and sparse normalizers; retain isolated types."""
    t,np,dgl,Sparse,native=(rt[k] for k in ('torch','numpy','dgl','SparseTensor','native'))
    adjs,edges,features={},{},{}
    with costs.measure('full_raw_features_and_pre_normalization_support_'+str(family)):
        for i,symbol in enumerate(SYMBOLS):
            features[symbol]=(t.eye(data.counts[i]) if data.features[i] is None else t.FloatTensor(np.frombuffer(data.features[i],dtype=np.float32).reshape(data.counts[i],-1)))
        for meta in data.relation_rows:
            h,k=meta['raw_head_type'],meta['raw_tail_type'];name=SYMBOLS[h]+SYMBOLS[k]
            pairs=() if name in REMOVAL[family] else data.canonical_relation_pairs[meta['raw_relation_id']]
            rows=t.LongTensor([head-data.offsets[h] for head,tail in pairs]);cols=t.LongTensor([tail-data.offsets[k] for head,tail in pairs])
            a=Sparse(row=rows,col=cols,sparse_sizes=(data.counts[h],data.counts[k]));adjs[name]=a
            edges[(SYMBOLS[k],SYMBOLS[k]+'-'+SYMBOLS[h],SYMBOLS[h])]=(cols.numpy(),rows.numpy())
            a.storage._value=None;a.storage._value=t.ones(a.nnz())/a.sum(dim=-1)[a.storage.row()]
        graph=dgl.heterograph(edges,num_nodes_dict={s:data.counts[i] for i,s in enumerate(SYMBOLS)})
        for symbol,value in features.items():graph.nodes[symbol].data[symbol]=value
        if any(graph.num_nodes(s)!=data.counts[i] for i,s in enumerate(SYMBOLS)):raise ValueError('All original typed nodes must remain')
    with costs.measure('all25_native_four_hop_feature_channels_'+str(family)):
        graph=native.hg_propagate_feat_dgl(graph,'M',4,5,[],echo=False)
        feats={k:graph.nodes['M'].data.pop(k).detach().clone() for k in list(graph.nodes['M'].data.keys())}
        if sorted(feats)!=rt['protocol']['feature_paths'] or len(feats)!=25:raise ValueError('Complete native feature channels required')
        if any(not t.isfinite(x).all().item() or x.shape[0]!=4932 for x in feats.values()):raise ValueError('Finite full native channel geometry required')
        removed_symbol=None if family is None else {'actor':'A','director':'D','keyword':'K'}[family]
        if removed_symbol and any(t.count_nonzero(v).item()!=0 for k,v in feats.items() if removed_symbol in k):raise ValueError('Removed raw family must produce empty dependent channels')
    graph=features=None;gc.collect()
    with costs.measure('all12_native_diagonal_removed_label_return_products_'+str(family)):
        products=native.hg_propagate_sparse_pyg(adjs,'M',4,5,[],prop_feats=False,echo=False,prop_device='cpu')
        if sorted(products)!=rt['protocol']['label_paths'] or len(products)!=12:raise ValueError('Complete native label return products required')
        products={k:rt['remove_diag'](v) for k,v in products.items()}
        if any(tuple(v.sizes())!=(4932,4932) for v in products.values()):raise ValueError('Native movie return geometry required')
    adjs=None;gc.collect();return feats,products


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized one-GPU route required before project operations')
    root=Path(args.project_root).resolve();out=root/args.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined independent typed custody required')
    os.environ['CUDA_VISIBLE_DEVICES']='';out.mkdir();(out/'features').mkdir();(out/'sealed').mkdir();costs=Costs(out)
    try:
        sources=bind_source(root);raw=root/'hgb_IMDB_public_acquisition_20261009_v1/development_inputs'
        bindings={n:receipt(root,raw/n) for n in RAW_FILES}
        if any(bindings[n]['bytes']!=v[0] or bindings[n]['sha256']!=v[1] for n,v in RAW_FILES.items()):raise ValueError('Exact complete raw dataset bytes changed')
        loader=load(root,sources['modules']['role_loader'],'data_loader');spec=json.loads(bound(root,sources['metadata']['schema']).read_text())
        with costs.measure('new_source_custody_full_typed_raw_schema_parse'):
            counts,offsets,features=loader.nodes(raw/'node.dat',spec);relations,pairs=loader.links(raw/'link.dat',counts,offsets,spec)
            labels=loader.development_labels(raw/'label.dat',counts[0],5)
        rt,engine=native_runtime(root,sources,'cpu');t,np=rt['torch'],rt['numpy'];t.set_num_threads(4)
        ids=sorted(labels);data=SimpleNamespace(counts=tuple(counts),offsets=tuple(offsets),features=tuple(features),relation_rows=tuple(relations),canonical_relation_pairs=pairs,class_dim=5,input_bindings=bindings)
        splits=[];file_paths=[];source_role_rows={}
        for split in range(10):
            shuffled=np.asarray(ids,dtype=np.int64);np.random.RandomState(99701+1009*split).shuffle(shuffled)
            n=len(shuffled);a=n//2;b=a+n//5;c=b+n//10
            roles={k:sorted(v.tolist()) for k,v in zip(('train','select','cal','evaluate'),(shuffled[:a],shuffled[a:b],shuffled[b:c],shuffled[c:]))}
            if set.union(*(set(v) for v in roles.values()))!=set(ids) or sum(len(v) for v in roles.values())!=len(ids):raise ValueError('Four disjoint complete development roles required')
            if any(not any(labels[i][j] for i in v) or not any(not labels[i][j] for i in v) for v in roles.values() for j in range(5)):raise ValueError('Every frozen role must contain both binary values for each label; do not silently resample')
            folder=out/('split_'+str(split));folder.mkdir();node_inputs={};endpoint_inputs={}
            for role,v in roles.items():
                path=(folder/(role+'.pt') if role in ('train','select') else out/'sealed'/('split_'+str(split)+'_'+role+'.pt'))
                t.save({'ids':t.LongTensor(v),'y':t.FloatTensor([labels[i] for i in v])},path);file_paths.append(path)
                if role in ('train','select'):node_inputs[role]=str(path.relative_to(out))
                else:
                    endpoint_inputs[role+'_targets']=str(path.relative_to(out));path=folder/(role+'_ids.pt');t.save(t.LongTensor(v),path);file_paths.append(path);endpoint_inputs[role+'_ids']=str(path.relative_to(out))
            role_descriptor={'schema':'typed-HGB-IMDB-four-role-development-v1','split_id':split,'role_seed':99701+1009*split,'rule':ROLE_RULE,'roles':roles,'input_files':bindings}
            path=folder/'ROLE.json';path.write_text(json.dumps(role_descriptor,indent=2,sort_keys=True)+'\n');file_paths.append(path)
            source_role_rows[split]=roles['train']
            splits.append({'split_id':split,'role_seed':99701+1009*split,'roles':{r:len(v) for r,v in roles.items()},'role_descriptor':str(path.relative_to(out)),
                'node_inputs':node_inputs,'endpoint_inputs':endpoint_inputs,'label_caches':{},'evaluate_purpose':'development'})
        graph_receipts={};feature_shapes=None
        for family in FAMILIES:
            name='full' if family is None else family;feats,products=typed_static(rt,data,family,costs)
            path=out/'features'/(name+'.pt')
            with costs.measure('native_source_feature_cache_write_'+name):t.save(feats,path)
            file_paths.append(path);graph_receipts[name]=str(path.relative_to(out));current={k:list(v.shape) for k,v in feats.items()}
            if feature_shapes is None:feature_shapes=current
            elif current!=feature_shapes:raise ValueError('Every source view preserves all native feature key shapes')
            for split,row in enumerate(splits):
                with costs.measure('native_TRAIN_only_label_product_and_write_'+name+'_split'+str(split)):
                    target=t.zeros(4932,5);train=source_role_rows[split];target[t.LongTensor(train)]=t.FloatTensor([labels[i] for i in train])
                    label_feats={k:v@target for k,v in products.items()}
                    if any(not t.isfinite(v).all().item() for v in label_feats.values()):raise ValueError('Finite native TRAIN-only label caches required')
                    path=out/('split_'+str(split))/(name+'_label_channels.pt');t.save(label_feats,path);file_paths.append(path);row['label_caches'][name]=str(path.relative_to(out))
                    target=label_feats=None
            feats=products=None;gc.collect();print(json.dumps({'prepared_native_graph_view':name,'complete_masks':10,'fits':0,'CUDA':False}),flush=True)
        if t.cuda.is_initialized():raise ValueError('CPU custody preparation initialized CUDA')
        file_rows=[{**receipt(out,path)} for path in file_paths]
        custody={'schema':'typed-HGB-IMDB-native-source-custody-v1','status':'prepared','dataset_id':'hgb-imdb-native5-development','new_to_original_paper':True,
            'historical_authorization':'unresolved and not inherited; current raw bytes independently checked on authorized route',
            'prior_evaluation_exposure':'encountered development; no unused-confirmation or official-test claim','official_archive_authority_verified':False,
            'raw_input_files':bindings,'source':sources,'node_counts':counts,'class_dim':5,'feature_channels':25,'label_channels':12,'feature_shapes':feature_shapes,
            'graph_feature_caches':graph_receipts,'splits':splits,'files':file_rows,'label_context':'TRAIN only; zero everywhere else; diagonal removed without renormalization',
            'preprocessing':'Exact native hg_propagate_feat_dgl/hg_propagate_sparse_pyg; raw paired family deletion before native means; all25/12 channels and isolated types retained',
            'feature_cache_reuse':'Four graph-only caches shared across every declared role/fit; full setup/materialization cost paid once here; no old cache/teacher reuse',
            'costs':costs.rows,'benchmark_fits':0,'predictive_scores':0,'native_model_constructor_calls':0,'native_resource_qualification':False,
            'CAL_evaluate_target_reads':'Only this preparation custodian writes them; source adapter must use TRAIN/SELECT and feature caches exclusively until phase gates',
            'TEST_file_open_stat_hash_or_parse':False,'CUDA':False,'GO':False}
        (out/'CUSTODY.json').write_text(json.dumps(custody,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'prepared','custody':receipt(root,out/'CUSTODY.json'),'files':len(file_rows),'fits':0,'CUDA':False,'GO':False}))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'fits':0,'GO':False,'costs':costs.rows},indent=2)+'\n');raise


if __name__=='__main__':main()
