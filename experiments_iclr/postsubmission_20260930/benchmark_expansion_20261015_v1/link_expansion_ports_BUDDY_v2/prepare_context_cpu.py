"""Materialize authentic native-year2007 BUDDY TRAIN/VALID inputs; zero updates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
import traceback


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def receipt(path):
    path=Path(path).resolve();h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
    return {'path':str(path),'bytes':size,'sha256':h.hexdigest()}


def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root).resolve();output=Path(args.output).resolve();package=Path(__file__).resolve().parent;expansion=package.parent;phase=expansion.parent
    assert output.is_relative_to(root) and package.is_relative_to(root) and not output.exists()
    output.mkdir(parents=True);resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(1800,1810))
    started=time.perf_counter();report={'schema':'native-BUDDY-year2007-materialized-context-v1','status':'running','stage':'before_import',
        'native_model_constructions':0,'optimizer_transitions':0,'model_forwards':0,'TEST_accessed':False,'device':'cpu','threads':2,
        'source':receipt(__file__),'memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()}
    def write():
        usage=resource.getrusage(resource.RUSAGE_SELF);report.update(elapsed_seconds=time.perf_counter()-started,process_peak_RSS_KiB=usage.ru_maxrss,process_user_seconds=usage.ru_utime,process_system_seconds=usage.ru_stime)
        temporary=output/'CONTEXT_MANIFEST.json.part';temporary.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');temporary.replace(output/'CONTEXT_MANIFEST.json')
    write();base=None
    try:
        import numpy as np
        import torch
        import datasketch
        import importlib.metadata
        torch.set_num_threads(2);torch.set_num_interop_threads(1);random.seed(0);np.random.seed(0);torch.random.default_generator.manual_seed(0)
        source_manifest=phase/'reference_wrapper_redesign_20261013_v1/smoke_20261014/MANIFEST.json'
        case=next(row for row in json.loads(source_manifest.read_text())['cases'] if row['id']=='collab-buddy/gnnm_boundary_native')
        recipe=case['recipe'];recipe['_repository_root']=str(root);recipe['_seed']=0;recipe['seed']=0
        if recipe['native_fixed_config']['year']!=2007:raise ValueError('Exact native year2007 recipe required')
        source=load(expansion/'link_native_integration_v1/session.py','_BUDDY_materialize_native_bridge')
        report.update(stage='native_source_construction',recipe=recipe,data_root=case['data_root'],smoke_manifest=receipt(source_manifest),
            bridge=receipt(expansion/'link_native_integration_v1/session.py'),datasketch_version=importlib.metadata.version('datasketch'),
            datasketch_module=receipt(datasketch.__file__),torch=str(torch.__version__))
        write();base=source.create_session(recipe,'gnnm_boundary_native',case['data_root'],'cpu');n=base.native;report['native_model_constructions']=1
        if base.family!='BUDDY' or n.args.year!=2007 or n.args.sign_k!=0:raise ValueError('Native BUDDY source configuration differs')
        def binding(value):
            raw=value.detach().contiguous().cpu().numpy()
            return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(memoryview(raw).cast('B')).hexdigest()}
        def materialize(links,labels=None,ids=None):
            values=tuple(value.detach().cpu().clone() for value in n._hydrate(links))
            if any(not torch.isfinite(value).all() for value in values):raise ValueError('Nonfinite authentic BUDDY input')
            result={'links':links.cpu().clone(),'inputs':values}
            if labels is not None:result['labels']=labels.cpu().clone()
            if ids is not None:result['sampler_ids']=ids.cpu().clone()
            return result
        report.update(stage='materialize_two_original_TRAIN_batches',source_TRAIN_positive_queries=len(n.positive),source_TRAIN_negative_queries=len(n.train_negative),
            native_binding=base.native_binding,original_optimizer_options=[{k:v for k,v in g.items() if k!='params'} for g in n.optimizer.param_groups])
        write();batches=[]
        for _ in range(2):
            ids=n._next_batch();batches.append(materialize(n.links[ids],n.labels[ids],ids))
        report['TRAIN_batches']=[{'records':len(row['links']),'positive':int((row['labels']==1).sum()),'negative':int((row['labels']==0).sum()),
            'links':binding(row['links']),'sampler_ids':binding(row['sampler_ids']),'inputs':[binding(value) for value in row['inputs']]} for row in batches]
        if any(row['records']!=1024 for row in report['TRAIN_batches']):raise ValueError('Keep native full1024 minibatches')
        report['stage']='materialize_complete_source_VALID';write()
        valid={role:materialize(n.valid[key]) for role,key in (('positive','edge'),('negative','edge_neg'))}
        if (len(valid['positive']['links']),len(valid['negative']['links']))!=(60084,100000):raise ValueError('Full official VALID roles required')
        common={'features':n.features.cpu().clone(),'degrees':n.degrees.cpu().clone(),'hash_tables':n.hash_tables,'cards':n.cards.cpu().clone(),
            'source_positive_queries':n.positive.cpu().clone(),'source_initial_negative_queries':n.train_negative.cpu().clone()}
        if not torch.isfinite(common['features']).all() or not torch.isfinite(common['cards']).all():raise ValueError('Nonfinite native features/hash cardinalities')
        static={name:binding(common[name]) for name in ('features','degrees','cards','source_positive_queries')}
        static['hash_tables']={str(hop):{name:binding(value) for name,value in table.items()} for hop,table in common['hash_tables'].items()}
        report.update(stage='serialize_authentic_native_context',static_tensor_bindings=static,
            full_VALID_inputs={key:{'links':binding(row['links']),'inputs':[binding(value) for value in row['inputs']]} for key,row in valid.items()},
            provider_data_files={role:receipt(path) for role,path in n.data_files.items()},
            training_reduction='one native BCE joint minibatch mean, no added class balancing',
            legacy_year0_cache_reused=False,source_model_class_replaced=False,source_sampler_or_hydration_replaced=False)
        write();path=output/'CONTEXT.pt';torch.save({'schema':'native-BUDDY-year2007-materialized-inputs-v1','seed':0,'recipe':recipe,'common':common,'TRAIN_batches':batches,'VALID_inputs':valid},path)
        report.update(context_file=receipt(path),status='passed',stage='complete',training_queries_materialized=2048,VALID_queries_materialized=160084,
            native_operators_unchanged=True,cache_semantics='source-year2007 weighted graph, source hash tables and native hydration; no old year0 cache values injected')
    except BaseException as exc:report.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(limit=14)})
    finally:
        if base is not None:base.close()
        report['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text();write()
    print(json.dumps({k:report.get(k) for k in ('status','stage','elapsed_seconds','optimizer_transitions','model_forwards','context_file')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
