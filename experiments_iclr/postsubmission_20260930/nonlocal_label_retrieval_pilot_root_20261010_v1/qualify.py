"""Real full-graph TRAIN derivative qualification, no validation scoring."""
from pathlib import Path
import hashlib,importlib.util,json,math,os,socket,subprocess,sys,time
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    frozen=json.loads((H/'QUALIFICATION_FREEZE.json').read_text())
    for row in frozen['repo_files']:
        assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    for row in frozen['bound_files']:
        assert hashlib.sha256((P/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    output=H/'actual_qualification_v1';output.mkdir(exist_ok=False)
    source=P/'nonlocal_train_label_retrieval_accuracy_hypothesis_20261010_v1/run_retrieval_family.py'
    spec=importlib.util.spec_from_file_location('fixed_retrieval_native',source)
    native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
    start=time.perf_counter();records=[]
    for backbone in ('SAGE','GCN','GAT'):
        cfgpath=H/backbone/'CONFIG.json';cfg=json.loads(cfgpath.read_text())
        scratch=output/backbone;scratch.mkdir()
        family=native.Family(cfg,scratch,hashlib.sha256(cfgpath.read_bytes()).hexdigest())
        torch=family.torch;torch.set_num_threads(2)
        labels=family.data.y[family.train_ids]
        assert min(int(labels.eq(c).sum()) for c in range(10))>=2
        for arm in ('live_shared4','detached_shared4','ordinary_M1_live'):
            family.current.update(arm=arm,seed=7301,member=0)
            bank=arm!='ordinary_M1_live'
            model,optimizer,streams=family.make(7301,0,'baseline',bank,4 if bank else 1)
            model.train();optimizer.zero_grad(set_to_none=True)
            generator=torch.Generator(device='cpu').manual_seed(7301+5000011)
            query=family.memory.common_half_query(torch,family.train_ids,labels,10,generator=generator)
            z,h=family.full_native(model,streams,bank)
            values=family.memory_read(z,h,query,'fit')
            ret=family.F.nll_loss(values['retrieval_log_probs'].flatten(0,1),family.data.y[query].repeat(len(z)))
            own=family.F.cross_entropy(z[:,family.train_ids].flatten(0,1),labels.repeat(len(z)))
            assert torch.isfinite(own) and torch.isfinite(ret)
            query_mask=torch.isin(family.train_ids,query)
            changed=labels.clone();changed[query_mask]=(changed[query_mask]+1)%10
            counter=family.memory.label_memory(torch,z,h,family.train_ids,changed,query,mode='fit',stop_attention_gradient=arm=='detached_shared4')
            assert torch.equal(values['support_rows'],counter['support_rows'])
            assert torch.allclose(values['retrieval_log_probs'],counter['retrieval_log_probs'],atol=1e-6,rtol=1e-6)
            if arm=='detached_shared4':
                assert not ret.requires_grad
                gradnorm=0.0
            else:
                g=torch.autograd.grad(ret,h,retain_graph=True)[0]
                assert torch.isfinite(g).all() and g[0,query].abs().sum()>0 and g[0,values['support_rows']].abs().sum()>0
                gradnorm=float(g.norm());del g
            total=own+ret;total.backward()
            gradients=[q.grad for q in model.parameters() if q.grad is not None]
            assert gradients and all(torch.isfinite(q).all() for q in gradients)
            assert any(q.abs().sum()>0 for q in gradients)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                eval_z,eval_h=family.full_native(model,streams,bank)
                served=family.memory_read(eval_z,eval_h,family.valid_ids[:32],'serve')['mixed_log_probs']
                assert served.shape==(4 if bank else 1,32,10) and torch.isfinite(served).all()
                assert torch.allclose(served.exp().sum(-1),torch.ones_like(served[...,0]),atol=1e-5,rtol=1e-5)
            records.append(dict(backbone=backbone,arm=arm,TRAIN_only_derivative=True,native_TRAIN_ce=float(own),retrieval_TRAIN_ce=float(ret),finite_gradients=True,hidden_retrieval_gradient_norm=gradnorm,common_query_label_input_exclusion=True,query_nodes=len(query),support_nodes=len(values['support_rows']),serving_shape_finite_normalized=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),validation_quality_read=False))
            del model,optimizer,streams,z,h,values,counter,ret,own,total,eval_z,eval_h,served,gradients
            torch.cuda.empty_cache()
        del family
    result=dict(qualified=True,cases=records,seconds=time.perf_counter()-start,fullgraph_forward_calls=18,native_member_trajectories=54,full_model_backwards=9,hidden_only_retrieval_VJPs=6,optimizer_updates=9,TRAIN_only=True,validation_accuracy_or_NLL_scored=False,TEST_access=False,scientific_quality_evidence=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output/'QUALIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'qualified':True,'cases':len(records),'seconds':result['seconds'],'fullgraph_forward_calls':18,'backwards':9,'scientific_quality_evidence':False}))
if __name__=='__main__':main()
