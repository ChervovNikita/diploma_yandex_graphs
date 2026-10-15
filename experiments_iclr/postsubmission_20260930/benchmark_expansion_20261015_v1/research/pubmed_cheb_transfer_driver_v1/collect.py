"""One genuine selected shared-bank VALID collection; zero TRAIN or selection."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--output',required=True);args = parser.parse_args()
    if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Dedicated allocation required before metadata reads')
    import study as s
    h = s.helpers(); root,here,output = Path(args.project_root).resolve(),Path(__file__).resolve().parent,Path(args.output).resolve()
    h.require(output.is_relative_to(root) and not output.exists() and Path(sys.prefix).resolve() == (root/'.venv').resolve(),
        'Original interpreter and fresh contained output required')
    h.require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0',h.GPU),'Exact original CUDA JOB required; no device substitution')
    spec = json.loads((here/'SPEC.json').read_text()); rows,_ = s.jobs(spec)
    a_path = h.safe(root,args.admission);h.require(h.pin(a_path)['sha256'] == args.admission_sha256,'Admission changed')
    a = json.loads(a_path.read_text())
    h.require(a.get('approved') is True and a.get('mode') == 'collect' and a.get('schema') == 'root-reviewed-pubmed-cheb-transfer-v1' and
        a.get('manifest_sha256') == h.pin(here/'MANIFEST.json')['sha256'] and
        a.get('SPEC_sha256') == h.pin(here/'SPEC.json')['sha256'] and a.get('TEST_access') is False,
        'Parent root collection admission required')
    for item in json.loads((here/'MANIFEST.json').read_text())['files']:
        h.verified(root,dict(item,path=str(here/item['path'])))
    study = h.safe(root,spec['study_output'])
    end_path,out_path = study/'END.json',study/'OUTCOMES.json'
    h.require(h.pin(end_path)['sha256'] == a['terminal_END_sha256'] and h.pin(out_path)['sha256'] == a['terminal_OUTCOMES_sha256'],
        'Actual all24 terminal metadata required')
    outcomes = json.loads(out_path.read_text());s.terminal_gate(json.loads(end_path.read_text()),outcomes,rows)
    h.require(0 <= args.index < 24 and rows[args.index]['kind'] != 'native','Only one of12 frozen shared banks may be collected')
    row = rows[args.index]
    h.require(output == h.safe(root,a['collection_output'])/row['token'],'Exact admitted collection destination required')
    fit,folder = s.fit(root,study,row,outcomes[args.index])
    sys.path.insert(0,str(s.E/'runtime'))
    from adapters import create_adapter
    from bookkeeping import Policy
    from common import seed_all,restore_rng,atomic_torch
    started = time.perf_counter();output.mkdir(parents=True)
    work = {'construction_attempts':0,'constructions_completed':0,'native_forward_attempts':0,
        'native_forwards_completed':0,'TRAIN_callbacks':0,'optimizer_steps':0}
    report = {'schema':'pubmed-cheb-own-selected-VALID-routes-v1','status':'running','role':'VALID','TEST_accessed':False,
        'job_id':row['id'],'job_hash':row['job_hash'],'cohort_seed':row['cohort_seed'],'kind':row['kind'],
        'selected_epoch':fit['selected_epoch'],'fit_RESULT':outcomes[args.index]['RESULT'],
        'selected_checkpoint':fit['selected_checkpoint'],'saved_VALID_export':fit['selected_VALID_export'],
        'admission':h.pin(a_path),'terminal_END':h.pin(end_path),'terminal_OUTCOMES':h.pin(out_path),
        'source':h.pin(__file__),'actual_work':work,'selector_calls':0,'selection_changed':False}
    adapter = None;hooks = [];original_update = None;steps = []
    signal.signal(signal.SIGTERM,h.interrupted)
    try:
        import torch
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();seed_all(torch,row['job']['seed'])
        selected = torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
        saved = torch.load(folder/'selected_VALID.pt',map_location='cpu',weights_only=False)
        s.accept_export(torch,saved,fit,row['job'],spec['provider'])
        h.require(selected['job_hash'] == row['job_hash'] and selected['runtime_binding'] == fit['runtime_binding'] and
            selected['selector']['selected_epoch'] == fit['selected_epoch'] and selected['TEST_accessed'] is False,
            'Own selected checkpoint/runtime authority differs')
        work['construction_attempts'] += 1
        adapter = create_adapter(torch,row['job'],Policy.from_dict(row['job']['policy'],'node'),root)
        work['constructions_completed'] += 1
        h.require(adapter.callback_receipt['sha256'] == fit['runtime_binding']['callback_sha256'] and
            (None if adapter.extension_receipt is None else adapter.extension_receipt['sha256']) == fit['runtime_binding']['extension_sha256'],
            'Actual native callback/composition differs')
        adapter.load_state(selected['adapter']);restore_rng(torch,selected['global_rng'])
        session,model = adapter.session,adapter.model
        original_update = session.train_update
        def forbidden_update(*args,**kwargs):work['TRAIN_callbacks'] += 1;raise ValueError('Collection cannot call TRAIN')
        session.train_update = forbidden_update
        for optimizer in adapter.optimizers:
            original = optimizer.step;steps.append((optimizer,original))
            def forbidden_step(*args,**kwargs):work['optimizer_steps'] += 1;raise ValueError('Collection cannot update Adam')
            optimizer.step = forbidden_step
        def before(module,args):
            work['native_forward_attempts'] += 1;h.require(work['native_forward_attempts'] <= 4,'Four VALID native forwards only')
        def after(module,args,result):work['native_forwards_completed'] += 1
        hooks = [model.native.register_forward_pre_hook(before),model.native.register_forward_hook(after)]
        report['construction_restore_seconds'] = time.perf_counter()-started
        before_forward = time.perf_counter();model.eval()
        with torch.no_grad():
            routes = session._routes(False)  # original evaluation route law, no activation-checkpoint training replay
            full = torch.stack([r.raw_logits for r in routes])
            ids = torch.nonzero(session.data.val_mask,as_tuple=False).flatten()
            labels = session.data.y[ids].detach().cpu()
            pooled = full.mean(0)[ids].detach().cpu()  # preserve original full-graph mean-before-mask operation
            members = full[:,ids].detach().cpu()
        torch.cuda.synchronize();report['VALID_forward_seconds'] = time.perf_counter()-before_forward
        h.require(len(routes) == 4 and tuple(members.shape) == (4,3943,3) and torch.equal(ids.cpu(),saved['ids']) and
            torch.equal(labels,saved['labels']) and torch.isfinite(members).all().item() and
            torch.isfinite(pooled).all().item() and work['native_forwards_completed'] == 4 and
            work['TRAIN_callbacks'] == work['optimizer_steps'] == 0,'Complete four-route aligned VALID with zero TRAIN required')
        difference = pooled.double()-saved['raw_logits'].double(); threshold = 1e-5+1e-4*saved['raw_logits'].double().abs()
        sanity = {'passed':bool(torch.allclose(pooled,saved['raw_logits'],rtol=1e-4,atol=1e-5)),
            'maximum_absolute_difference':float(difference.abs().max()),'outside_tolerance':int((difference.abs()>threshold).sum()),
            'rtol':1e-4,'atol':1e-5,'diagnostic_only_no_retry_or_scoring_substitution':True}
        raw = {'schema':'pubmed-cheb-selected-VALID-member-logits-v1','role':'VALID','TEST_accessed':False,
            'job_id':row['id'],'job_hash':row['job_hash'],'selected_epoch':fit['selected_epoch'],
            'selected_checkpoint':fit['selected_checkpoint'],'saved_VALID_export':fit['selected_VALID_export'],
            'ids':ids.cpu(),'labels':labels,'member_raw_logits':members,'reconstructed_pool_raw_logits':pooled,'sanity':sanity}
        receipt = atomic_torch(torch,output/'ROUTE_VALID.pt',raw)
        report.update(status='complete' if sanity['passed'] else 'complete_with_sanity_warning',
            members=4,VALID_nodes=3943,VALID_class_counts=[72,1919,1952],raw_tensor_receipt=receipt,sanity=sanity)
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=10)})
    finally:
        for hook in hooks:hook.remove()
        if adapter is not None:
            if original_update is not None:adapter.session.train_update = original_update
            for optimizer,original in steps:optimizer.step = original
            try:adapter.close()
            except Exception as error:report['close_error'] = str(error);report['status'] = 'failed'
        report.update(wall_seconds=time.perf_counter()-started,resources={'own_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
        if 'torch' in locals() and torch.cuda.is_initialized():report['resources'].update(
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        h.write(output/'RESULT.json',report)
    return 0 if report['status'] in ('complete','complete_with_sanity_warning') else 1


if __name__ == '__main__':raise SystemExit(main())
