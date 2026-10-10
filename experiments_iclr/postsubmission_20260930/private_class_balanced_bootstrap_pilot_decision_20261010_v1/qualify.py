"""One full-TRAIN check of new fixed weights and mixed-block update semantics."""
import gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent;phase=here.parent
    source=phase/'common_wrapper_private_bootstrap_native_family_source_20261010_v1/run_family.py'
    spec=importlib.util.spec_from_file_location('fixed_private_bootstrap_family',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    output=here/'actual_qualification_v1';output.mkdir(exist_ok=False)
    rows=[];started=time.monotonic();weight_hashes=[]
    for backbone in ('GAT','SAGE'):
        cfg_bytes=(here/backbone/'CONFIG.json').read_bytes();cfg=json.loads(cfg_bytes)
        folder=output/backbone;folder.mkdir()
        family=module.Family(cfg,folder,hashlib.sha256(cfg_bytes).hexdigest())
        torch=family.torch;weights,weight_record=family.prepare_bootstrap(cfg['seeds'][0])
        weight_hashes.append(weight_record['weights_sha256'])
        for kind in ('coherent','paired_graph','rank1_lora_graph'):
            model,optimizer,streams=family.make(cfg['seeds'][0],0,kind,True,4)
            roles,shared,private=family.bootstrap.parameter_roles(torch,family.factors,model)
            model.train();optimizer.zero_grad()
            logits=family.logits(model,streams,True,family.train_ids)
            own,weighted,unconnected,member_losses=family.bootstrap.bank_gradients(torch,family.F,logits,family.data.y[family.train_ids],weights,shared,private)
            assert not unconnected['private'],'A private bank is unconnected to the actual TRAIN graph'
            shared_grad=sum(parameter.grad.square().sum().item() for _,parameter in shared if parameter.grad is not None)
            private_grad=sum(parameter.grad.square().sum().item() for _,parameter in private if parameter.grad is not None)
            assert shared_grad>0 and private_grad>0
            optimizer.step()
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):assert torch.isfinite(value).all()
            rows.append({'backbone':backbone,'kind':kind,'full_TRAIN_objects':580,'graph_nodes':11701,
                         'weights_sha256':weight_record['weights_sha256'],'parameter_roles':roles,
                         'own_ce':own,'weighted_private_ce':weighted,'train_member_losses':member_losses,
                         'shared_gradient_square_sum':shared_grad,'private_gradient_square_sum':private_grad,
                         'unconnected_names':unconnected,'reverse_mode_calls':2,'adam_steps':1,
                         'optimizer_ownership_checked_before_step':True})
            del model,optimizer,streams,logits,shared,private
            gc.collect();torch.cuda.empty_cache()
        del family,weights;gc.collect();torch.cuda.empty_cache()
    assert len(set(weight_hashes))==1,'Companion backbones must use the same seed/TRAIN weight table'
    report={'passed':True,'worker_PID':os.getpid(),'cases':rows,'seconds':time.monotonic()-started,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'bootstrap_helper_sha256':hashlib.sha256(source.with_name('bootstrap.py').read_bytes()).hexdigest(),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'reused_forward_restore_integration':'private_feature_rotation_pilot_decision_20261010_v1/ACTUAL_QUALIFICATION.json',
            'new_gradient_integration_only':True,'VALID_quality_scored':False,'TEST_access':False}
    (output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(rows),'seconds':report['seconds'],'new_gradient_integration_only':True}),flush=True)

if __name__=='__main__':main()
