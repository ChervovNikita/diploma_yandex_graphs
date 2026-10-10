"""Nine disposable native TRAIN units covering all fixed learning conditions."""
from pathlib import Path
import hashlib, importlib.util, json, socket, subprocess, sys, time
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    frozen=json.loads((H/'QUALIFICATION_FREEZE.json').read_text())
    for row in frozen['bound_files']:
        assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    out=H/'actual_qualification_v1';out.mkdir(exist_ok=False)
    source=P/'native_SAGE_GNCL_SupCon_2x2_source_20261010_v1/run_family.py'
    spec=importlib.util.spec_from_file_location('sage_factorial_qualification',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=H/'CONFIG.json';family=module.Family(json.loads(config.read_text()),out,hashlib.sha256(config.read_bytes()).hexdigest())
    torch=family.torch;torch.set_num_threads(2)
    def forbidden(*args,**kwargs):raise RuntimeError('No VALID scoring during disposable TRAIN qualification')
    family.metrics=forbidden
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();cases=[];units=[];trajectories=0;vjp_count=0
    for arm in module.ARMS:
        bank=arm.startswith('shared4_');members=4 if bank else 1
        use_sc=arm!='shared4_GNCL';objective='gncl_half' if arm in ('shared4_GNCL','shared4_GNCL_SupCon') else 'own_only'
        independent=[]
        for member in range(4 if arm=='genuine_factorized_I4_SupCon' else 1):
            model,optimizer,streams=family.make(7301,member,'baseline',arm!='ordinary_M1_SupCon',members)
            if arm=='genuine_factorized_I4_SupCon':
                identities={id(q) for q in model.parameters()}
                assert all(not identities.intersection(old) for old in independent)
                independent.append(identities);units.append((model,optimizer,streams))
            model.train();optimizer.zero_grad(set_to_none=True)
            logits,hidden=family.training_outputs(model,streams,bank);trajectories+=members
            targets=family.data.y[family.train_ids] if bank else family.data.y[family.data.train_mask]
            assert tuple(logits.shape)==(members,580,10) and tuple(hidden.shape)==(members,580,128)
            assert hidden.requires_grad and torch.isfinite(logits).all() and torch.isfinite(hidden).all()
            own=family.F.cross_entropy(logits.flatten(0,1),targets.repeat(members));pool=family.probability_pool_ce(logits,targets)
            sc,rsc=family.within_route_supcon(hidden,targets) if use_sc else (None,None)
            loss=.5*own+.5*pool if objective=='gncl_half' else own
            if use_sc:loss=loss+.05*sc
            assert all(torch.isfinite(t).item() for t in [own,pool,loss]+([sc] if use_sc else []))
            gradients=None
            if use_sc:
                named=list(model.named_parameters());g=torch.autograd.grad(sc,[q for n,q in named],retain_graph=True,allow_unused=True);vjp_count+=1
                assert all(t is None or torch.isfinite(t).all() for t in g)
                head=[t for (n,q),t in zip(named,g) if 'output_linear.' in n]
                assert head and all(t is None for t in head),'SupCon must stop before the classifier'
                upstream=[float(t.abs().sum()) for (n,q),t in zip(named,g) if 'output_linear.' not in n and t is not None]
                assert upstream and sum(upstream)>0
                private=[float(t.abs().sum()) for (n,q),t in zip(named,g) if 'output_linear.' not in n and n.endswith(('.r','.s')) and t is not None]
                if arm!='ordinary_M1_SupCon':assert private and sum(private)>0
                gradients=dict(upstream_absolute_sum=sum(upstream),private_absolute_sum=sum(private),output_head_SC_gradients_absent=True)
            loss.backward()
            assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
            assert any(q.grad is not None and q.grad.abs().sum().item()>0 for q in model.parameters())
            optimizer.step()
            cases.append(dict(arm=arm,member=member,routes=members,own_CE=float(own.detach()),pool_CE=float(pool.detach()),SC=float(sc.detach()) if sc is not None else None,total_loss=float(loss.detach()),logits_shape=list(logits.shape),H_shape=list(hidden.shape),SC_gradients=gradients,all_objective_gradients_finite=True))
            del logits,hidden,targets,own,pool,sc,rsc,loss
            if use_sc:del named,g,head,upstream,private
            if arm!='genuine_factorized_I4_SupCon':del model,optimizer,streams
    torch.cuda.synchronize()
    result=dict(qualified=True,cases=cases,seconds=time.perf_counter()-start,full_native_forwards=9,native_member_trajectories=trajectories,full_parameter_backwards=9,SC_VJPs=vjp_count,discarded_optimizer_updates=9,scientific_fits=0,genuine_I4_four_disjoint_parameter_instances=True,TRAIN_class_counts=torch.bincount(family.data.y[family.train_ids],minlength=10).cpu().tolist(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),VALID_labels_loaded_only_by_existing_role_loader=True,VALID_quality_scored=False,TEST_access=False,scientific_quality_evidence=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert trajectories==18 and result['cuda_peak_allocated_bytes']<32*1024**3
    (out/'QUALIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(qualified=True,cases=len(cases),seconds=result['seconds'],peak_GiB=result['cuda_peak_allocated_bytes']/1024**3,scientific_quality_evidence=False)))
if __name__=='__main__':main()
