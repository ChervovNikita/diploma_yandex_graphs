"""One full-TRAIN operator integration; no prediction-quality scoring."""
import gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

def load(path):
    spec=importlib.util.spec_from_file_location("rotation_native_family_qualifier",path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def main():
    here=Path(__file__).resolve().parent
    repo=here.parents[2]
    phase=here.parent
    output=here/'actual_qualification_v1';output.mkdir(exist_ok=False)
    source=phase/'common_wrapper_paired_graph_native_family_source_20261010_v1/run_family.py'
    m=load(source);rows=[];started=time.monotonic()
    for backbone in ('GAT','SAGE'):
        cfg_bytes=(here/backbone/'CONFIG.json').read_bytes();cfg=json.loads(cfg_bytes)
        folder=output/backbone;folder.mkdir()
        f=m.Family(cfg,folder,hashlib.sha256(cfg_bytes).hexdigest());t=f.torch
        reference=None;extra_counts={}
        for kind in ('coherent','paired_graph','rank1_lora_graph','paired_local'):
            model,opt,streams=f.make(cfg['seeds'][0],0,kind,True,4)
            model.eval()
            with t.no_grad():before=f.logits(model,streams,True,f.train_ids).detach()
            if reference is None:reference=before.clone()
            else:t.testing.assert_close(before,reference,rtol=2e-4,atol=2e-5)
            extras={name:v for name,v in model.named_parameters() if name.endswith(('.u1','.u2','.lora_a','.lora_b'))}
            extra_counts[kind]=sum(v.numel() for v in extras.values())
            for module in model.modules():
                if hasattr(module,'u1'):
                    assert module.u1 is not module.u2 and module.u1.data_ptr()!=module.u2.data_ptr()
                    assert t.equal(module.u1,module.u2)
            model.train();opt.zero_grad()
            logits=f.logits(model,streams,True,f.train_ids)
            loss=f.F.cross_entropy(logits.flatten(0,1),f.data.y[f.train_ids].repeat(4))
            assert t.isfinite(loss)
            loss.backward()
            grad_sums={}
            for name,v in extras.items():
                assert v.grad is not None and t.isfinite(v.grad).all()
                value=v.grad.abs().sum().item();grad_sums[name]=value
                if not name.endswith('.lora_a'):assert value>0,name
            opt.step()
            assert all(t.isfinite(v).all() for v in model.parameters())
            model.eval()
            with t.no_grad():after=f.logits(model,streams,True,f.train_ids).detach()
            checkpoint=folder/(kind+'.pt')
            t.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'streams':streams},checkpoint)
            saved=t.load(checkpoint,map_location='cpu',weights_only=True)
            with t.no_grad():next(model.parameters()).zero_()
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);streams=saved['streams']
            model.check_optimizer_ownership(opt)
            with t.no_grad():restored=f.logits(model,streams,True,f.train_ids)
            t.testing.assert_close(restored,after,rtol=2e-4,atol=2e-5)
            roles=model.parameter_roles()
            # Existing role reporter does not know extra banks; report them explicitly.
            roles['private_correction']=list(extras)
            roles['shared_block']=[n for n in roles['shared_block'] if n not in extras]
            rows.append({'backbone':backbone,'kind':kind,'full_TRAIN_objects':580,'graph_nodes':11701,
                         'initial_function_max_abs':(before-reference).abs().max().item(),
                         'extra_private_parameters':extra_counts[kind],'correction_sites':model._correction_sites,
                         'extra_gradient_abs_sums':grad_sums,'parameter_roles':roles,
                         'updates':1,'backwards':1,'adam_steps':1,'strict_checkpoint_restore':True})
            del model,opt,streams,before,after,restored,logits,extras,saved
            gc.collect();t.cuda.empty_cache()
        assert extra_counts['coherent']==0
        assert extra_counts['paired_graph']==extra_counts['rank1_lora_graph']==extra_counts['paired_local']==(2048 if backbone=='GAT' else 4096)
        del reference,f;gc.collect();t.cuda.empty_cache()
    report={'passed':True,'worker_PID':os.getpid(),'cases':rows,'seconds':time.monotonic()-started,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'corrections_sha256':hashlib.sha256(source.with_name('corrections.py').read_bytes()).hexdigest(),
            'engineering_only':True,'VALID_quality_scored':False,'TEST_access':False}
    (output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(rows),'seconds':report['seconds'],'engineering_only':True}),flush=True)

if __name__=='__main__':main()
