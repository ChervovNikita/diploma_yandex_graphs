"""Actual full SAGE TRAIN qualification; no validation score or fit family."""
from pathlib import Path
import hashlib, importlib.util, json, os, socket, subprocess, sys, time
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
    out=H/'actual_qualification_v1';out.mkdir(exist_ok=False)
    source=P/'shared_recurrent_class_decoder_native_source_20261010_v1/native_extension.py'
    spec=importlib.util.spec_from_file_location('actual_class_decoder_source',source)
    extension=importlib.util.module_from_spec(spec);sys.modules[spec.name]=extension;spec.loader.exec_module(extension)
    cfg=json.loads((H/'CONFIG.json').read_text())
    family=extension.family_class()(cfg,out,hashlib.sha256((H/'CONFIG.json').read_bytes()).hexdigest())
    torch=family.torch;torch.set_num_threads(2)
    def forbidden_metrics(*args,**kwargs):raise RuntimeError('No VALID scoring is admitted in qualification')
    family.metrics=forbidden_metrics
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();records=[]
    for arm in extension.ARMS:
        bank=arm in extension.BANK_ARMS
        members=4 if bank else 1
        family.current.update(arm=arm,seed=7301,member=0)
        model,optimizer,streams=family.make(7301,0,'baseline',arm!=extension.ARMS[0],members)
        model.train();optimizer.zero_grad(set_to_none=True)
        native,scores=family.predictions(model,streams,bank)
        assert torch.equal(native,scores), 'Zero-B scores must be exactly native'
        labels=family.data.y[family.train_ids].repeat(members)
        own=family.F.cross_entropy(native[:,family.train_ids].flatten(0,1),labels)
        decoded=family.F.cross_entropy(scores[:,family.train_ids].flatten(0,1),labels)
        loss=.5*own+.5*decoded
        g_own=torch.autograd.grad(own,native,retain_graph=True)[0]
        g_total=torch.autograd.grad(loss,native,retain_graph=True)[0]
        assert torch.allclose(g_own,g_total,atol=1e-6,rtol=1e-5)
        loss.backward()
        gradients=[v.grad for v in model.parameters()if v.grad is not None]
        assert gradients and all(torch.isfinite(g).all()for g in gradients)
        B=model.class_decoder_B
        assert torch.isfinite(B.grad).all()and B.grad.abs().sum()>0
        if arm==extension.ARMS[3]:
            parameter_ids=[{id(v)for v in body.parameters()}for body in model]
            assert all(not(parameter_ids[i]&parameter_ids[j])for i in range(4)for j in range(i))
        optimizer.step();model.eval()
        z,d=family.predictions(model,streams,bank)
        ce=family.F.cross_entropy(d[:,family.train_ids].flatten(0,1),labels)
        vjp=torch.autograd.grad(ce,z)[0]
        mask=torch.ones(len(family.data.x),device=family.device,dtype=torch.bool);mask[family.train_ids]=False
        neighbor_gradient=float(vjp[:,mask].abs().sum())
        assert (neighbor_gradient==0)if arm==extension.ARMS[-1]else(neighbor_gradient>0)
        assert torch.isfinite(z).all()and torch.isfinite(d).all()and torch.isfinite(vjp).all()
        records.append({'arm':arm,'members':members,'zero_B_native_identity':True,'initial_native_gradient_identity':True,'finite_parameter_gradients':True,'decoder_gradient_norm':float(B.grad.norm()),'off_TRAIN_native_logit_gradient_sum_after_update':neighbor_gradient,'untied_storage_checked':arm==extension.ARMS[3],'parameters':sum(v.numel()for v in model.parameters()),'decoder_parameters':B.numel(),'TRAIN_loss':float(loss),'semantics':family.arm_metadata(arm)})
        del model,optimizer,streams,native,scores,own,decoded,loss,g_own,g_total,gradients,B,z,d,ce,vjp,mask
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    v={'qualified':True,'cases':records,'seconds':time.perf_counter()-start,'full_native_forwards':14,'native_member_trajectories':38,'full_parameter_backwards':7,'optimizer_updates':7,'native_logit_VJPs':21,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'VALID_labels_loaded_only_by_existing_role_loader':True,'VALID_quality_scored':False,'TEST_access':False,'scientific_quality_evidence':False,'decoder_graph':family.decoder_graph,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    assert v['cuda_peak_allocated_bytes']<32*1024**3
    (out/'QUALIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps({'qualified':True,'cases':len(records),'seconds':v['seconds'],'peak_GiB':v['cuda_peak_allocated_bytes']/1024**3,'scientific_quality_evidence':False}))
if __name__=='__main__':main()
