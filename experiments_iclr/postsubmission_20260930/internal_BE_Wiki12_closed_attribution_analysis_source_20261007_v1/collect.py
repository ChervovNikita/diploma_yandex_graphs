"""Disabled closed Wiki12 collector: original selected states, one pass/member."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
from gate import HERE, PHASE, SEEDS, MAX_FORWARDS, binding, bound, consume, inside, module, read, require


def write(path, value):
    import json
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'); os.replace(temporary,path)


def restore(torch, model, saved, record, config, specification, source_pins, public):
    require(isinstance(saved,dict) and saved.get('config') == config and saved.get('mechanism_ablation') == specification,
            'Original selected condition/config/adapter identity')
    run = saved['run']
    require(run['task'] == 'wikics' and run['arm'] == run['method_identity'] == record['method_identity']
            and run['seed'] == record['seed'] and run['mechanism_ablation'] == specification
            and run['ablation_adapter_sha256'] == specification['ablation_adapter_sha256']
            and run['execution_mode'] == specification['execution_mode'] and run['TEST_scoring'] is False,
            'Selected snapshot retains exact v2 method identity')
    require(run['data']['train_npz_sha256'] == source_pins['train']['sha256']
            and run['data']['valid_npz_sha256'] == source_pins['development']['sha256'], 'Selected role hashes')
    require(run['core'] == {name: binding(public.ROOT/'core'/name)['sha256'] for name in ('factors.py','models.py','objectives.py','selection.py')}
            and run['native']['polynormer_model_sha256'] == source_pins['polynormer']['sha256'], 'Selected core/native functions')
    require(type(saved.get('global')) is bool and type(saved.get('epoch')) is int and 1 <= saved['epoch'] <= 1100
            and saved.get('checkpoint_kind') == 'strict-first-maximum complete VALID joint snapshot', 'Original selected joint state/mode')
    model.load_state_dict(saved['model'],strict=True)
    model.set_global(saved['global'])  # Python flag is absent from state_dict; a selected local state stays local.
    streams = saved['streams']
    require(len(streams) == 4 and all(set(s)=={'cpu','cuda'} and all(t.device.type=='cpu' and t.dtype==torch.uint8 for t in s.values()) for s in streams), 'Saved CPU byte member RNG streams')
    require(len(saved['member_VALID']) == 4 and all(math.isfinite(x) for x in saved['member_VALID'])
            and math.isfinite(saved['selected_VALID']), 'Original selected scalar metadata finite')
    model.eval()
    return streams, dict(epoch=saved['epoch'], global_mode=saved['global'],
        stored_accuracy=saved['selected_VALID'], stored_member_accuracy=saved['member_VALID'],
        method_identity=record['method_identity'], reselection=False, bitwise_parity_claimed=False)


def collect_cell(torch,np,public,adapter,analysis,contract,sources,batch,truth,record,source_pins,output,cost):
    started=time.monotonic(); model=None; saved=None; completed_logits=[]; completed_representations=[]
    partial=output/'raw'/(record['cell']+'_partial.npz'); record.update(collection_status='running',attempted_member_forwards=0,completed_member_forwards=0)
    torch.cuda.reset_peak_memory_stats(); phase='construction'; phase_started=started; persistence=0.
    try:
        seed,condition=record['seed'],record['condition']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        config=adapter.configured_recipe(public,condition); spec=adapter.identity(condition)
        require(spec['control_id']==record['method_identity'],'V2 condition label must not become ordinary be_unit')
        model=public._core()['models'].Ensemble('wikics',spec['underlying_session_arm'],seed,config['model'],sources).to('cuda:0')
        require(model.members==4 and not model.independent,'Original shared4 unit-factor model')
        torch.cuda.synchronize(); record['construction_seconds']=time.monotonic()-phase_started
        phase='checkpoint_load_and_restore'; phase_started=time.monotonic(); checkpoint=bound(record['selected_checkpoint'])
        cost['checkpoint_deserialization_attempts']+=1; cost['checkpoint_bytes_submitted_to_deserializer']+=record['selected_checkpoint']['bytes']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        streams,metadata=restore(torch,model,saved,record,config,spec,source_pins,public)
        bound(record['selected_checkpoint']); public._core()['selection'].finite_state(model,[])
        record['selected_metadata']=metadata; del saved; saved=None
        torch.cuda.synchronize(); record['checkpoint_load_and_restore_seconds']=time.monotonic()-phase_started
        phase='inference_and_CPU_transfer'; phase_started=time.monotonic()
        with torch.no_grad():
            for member in range(4):
                require(cost['attempted_member_forwards']<MAX_FORWARDS,'Fixed48 inference budget')
                cost['attempted_member_forwards']+=1; record['attempted_member_forwards']+=1
                serial=time.monotonic(); write(output/'compact'/'PROGRESS.json',dict(cell=record['cell'],member_index0=member,
                    stage='call_about_to_start',attempted_member_forwards=cost['attempted_member_forwards'],completed_member_forwards=cost['completed_member_forwards'],automatic_retry=False))
                persistence+=time.monotonic()-serial; call_started=time.monotonic()
                with torch.random.fork_rng(devices=[0]):
                    torch.set_rng_state(streams[member]['cpu']); torch.cuda.set_rng_state(streams[member]['cuda'],0)
                    logits,representations=model.member_forward(batch,member)
                torch.cuda.synchronize()
                cost['completed_member_forwards']+=1; record['completed_member_forwards']+=1
                require(tuple(logits.shape)==(5274,10) and tuple(representations.shape)==(5274,512)
                        and torch.isfinite(logits).all() and torch.isfinite(representations).all(),'Finite complete original member inference')
                completed_logits.append(logits.cpu().numpy().copy()); completed_representations.append(representations.cpu().numpy().copy())
                torch.cuda.synchronize()
                record.setdefault('member_inference_and_transfer_seconds',[]).append(time.monotonic()-call_started)
                del logits,representations
                serial=time.monotonic(); np.savez(partial,member_logits=np.stack(completed_logits),member_representations=np.stack(completed_representations),valid_ids=batch['ids'].cpu().numpy(),truth=truth.numpy())
                record['partial_archive']=binding(partial); write(output/'compact'/'PROGRESS.json',dict(cell=record['cell'],member_index0=member,
                    stage='returned_and_partial_saved',attempted_member_forwards=cost['attempted_member_forwards'],completed_member_forwards=cost['completed_member_forwards'],partial_archive=record['partial_archive'],automatic_retry=False))
                persistence+=time.monotonic()-serial
            logits=torch.from_numpy(np.stack(completed_logits)).to('cuda:0')
            member_probability=logits.softmax(-1); pooled=member_probability.mean(0)
            public._core()['selection'].finite_predictions(logits,pooled)
            truth_gpu=truth.to('cuda:0'); target=torch.nn.functional.one_hot(truth_gpu,num_classes=10).to(logits.dtype)
            member_logp=logits.log_softmax(-1).gather(-1,truth_gpu[None,:,None].expand(4,-1,1)).squeeze(-1)
            member_nll=-member_logp; pool_nll=-(torch.logsumexp(member_logp,dim=0)-math.log(4))
            member_brier=(member_probability-target[None]).square().sum(-1); pool_brier=(pooled-target).square().sum(-1)
            logits_cpu=logits.cpu(); pooled_cpu=pooled.cpu(); predictions=logits_cpu.argmax(-1); pool_prediction=pooled_cpu.argmax(-1)
            record['source_float32_reevaluation']=dict(accuracy=float((pool_prediction==truth).float().mean()),members=[float((p==truth).float().mean()) for p in predictions])
            arrays=dict(valid_ids=batch['ids'].cpu().numpy().copy(),truth=truth.numpy().copy(),member_logits=logits_cpu.numpy().copy(),
                member_representations=np.stack(completed_representations),member_probability=member_probability.cpu().numpy().copy(),pool_probability=pooled_cpu.numpy().copy(),
                member_prediction=predictions.numpy().copy(),pool_prediction=pool_prediction.numpy().copy(),member_nll=member_nll.cpu().numpy().copy(),pool_nll=pool_nll.cpu().numpy().copy(),
                member_brier=member_brier.cpu().numpy().copy(),pool_brier=pool_brier.cpu().numpy().copy())
        record['inference_and_CPU_transfer_seconds']=time.monotonic()-phase_started-persistence
        analysis.validate(np,arrays,contract); phase='serialization'; phase_started=time.monotonic()
        archive=output/'raw'/(record['cell']+'.npz'); np.savez(archive,**arrays); record['raw_archive']=binding(archive)
        partial.unlink(); record.pop('partial_archive',None); record['serialization_seconds']=time.monotonic()-phase_started
        record['collection_status']='complete'
        if condition=='plain':
            phase_started=time.monotonic(); masks=contract.baseline_cohorts(np,arrays); path=output/'raw'/('plain_cohorts_'+str(seed)+'.npz'); np.savez(path,**masks)
            record['cohort_freeze_seconds']=time.monotonic()-phase_started
            return dict(available=True,plain_cell=record['cell'],plain_archive=record['raw_archive'],archive=binding(path),counts={name:int(masks[name].sum()) for name in contract.COHORTS},frozen_before_candidate_prediction_inspection=True)
    except Exception as error:
        record['collection_status']='retained_collection_failure'; record['failure']=dict(error_type=type(error).__name__,error=str(error),phase=phase,failed_phase_seconds=time.monotonic()-phase_started,automatic_retry=False)
    finally:
        record['inclusive_cell_seconds']=time.monotonic()-started; record['progress_and_partial_storage_seconds']=persistence
        record['peak_CUDA_allocated_bytes']=int(torch.cuda.max_memory_allocated()); record['peak_CUDA_reserved_bytes']=int(torch.cuda.max_memory_reserved())
        cost['peak_CUDA_allocated_bytes']=max(cost['peak_CUDA_allocated_bytes'],record['peak_CUDA_allocated_bytes']); cost['peak_CUDA_reserved_bytes']=max(cost['peak_CUDA_reserved_bytes'],record['peak_CUDA_reserved_bytes'])
        del model,saved; gc.collect(); torch.cuda.empty_cache()
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--release',type=Path,required=True); parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args(); os.umask(0o077); started=time.monotonic(); usage_start=resource.getrusage(resource.RUSAGE_SELF)
    cfg,pins,source_pins,records,output=consume(args.release,args.release_sha256)
    require(socket.gethostname()=='peptide' and Path.cwd().resolve()==Path(source_pins['repository']).resolve()
            and str(Path(sys.executable).absolute())==source_pins['python'] and os.environ.get('PYTHONPATH','')=='','Original GPU77 repo-owned runtime')
    require(cfg['physical_gpu_uuid']==source_pins['physical_gpu_inventory'][0] and os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['physical_gpu_uuid'],'Fixed analysis physical GPU0')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    require(rows==source_pins['physical_gpu_inventory'],'Original physical inventory')
    free_mib=int(subprocess.check_output(['nvidia-smi','--id='+cfg['physical_gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    require(free_mib*1048576>=cfg['minimum_fresh_GPU_free_bytes'],'Fresh source-reviewed analysis GPU headroom')
    output.mkdir(mode=0o700); (output/'raw').mkdir(mode=0o700); (output/'compact').mkdir(mode=0o700)
    collection=dict(schema='Wiki12-closed-attribution-predictions-v1',cells=records,cohorts={str(seed):dict(available=False) for seed in SEEDS},whole12_complete_before_opening=True,status='running',TEST_access=False,reselection=False,automatic_retry=False)
    cost=dict(schema='Wiki12-closed-attribution-analysis-cost-v1',status='failed',started_UTC=datetime.now(timezone.utc).isoformat(),release=binding(args.release),
        metadata_gate_seconds=time.monotonic()-started,maximum_member_forwards=MAX_FORWARDS,attempted_member_forwards=0,completed_member_forwards=0,
        checkpoint_deserialization_attempts=0,checkpoint_bytes_submitted_to_deserializer=0,peak_CUDA_allocated_bytes=0,peak_CUDA_reserved_bytes=0,
        fullgraph_nodes_per_call=11701,development_objects_per_call=5274,TRAIN_updates=0,backward_calls=0,optimizer_constructions=0,
        raw_logits_representations_and_labels_server_only=True,TEST_access=False)
    for record in records: record['collection_status']='pending'
    write(output/'compact'/'GATE.json',dict(passed=True,source= pins['training_source_manifest'],bundle=pins['scientific_bundle'],family_closure=cfg['family_closure'],lane_closures=cfg['lane_closures'],parent_owner=cfg['parent_owner'],terminal_evidence=cfg['terminal_evidence'],all12_original_selected_hashes_validated=True,TEST_access=False))
    try:
        # First numerical imports occur only after the entire closed-family gate.
        setup=time.monotonic(); import numpy as np; import torch
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        versions={'torch':str(torch.__version__),'numpy':np.__version__,**{name:importlib.metadata.version(name) for name in ('torch-geometric','torch-scatter','torch-sparse','ogb')}}
        require(versions=={'torch':'2.1.2+cu118','numpy':'1.26.4','torch-geometric':'2.7.0','torch-scatter':'2.1.2+pt21cu118','torch-sparse':'0.6.18+pt21cu118','ogb':'1.3.6'},'Original matched scientific runtime versions')
        public=module(bound(pins['public_adapter']),'_wiki12_original_public'); sys.modules['portable']=public
        data=module(bound(pins['public_data']),'_wiki12_original_data')
        recompute=module(bound(pins['recomputation']),'_wiki12_recomputation_identity_only'); sys.modules['recompute']=recompute
        adapter=module(bound(pins['training_adapter']),'_wiki12_original_condition_identity')
        analysis=module(HERE/'analyse.py','_wiki12_analysis'); contract=analysis.contracts(pins)
        sources,native_origin=public.native_sources('wikics',bound(source_pins['polynormer']))
        cost['source_and_runtime_loading_seconds']=time.monotonic()-setup; cost['runtime_versions']=versions; cost['native']=native_origin
        setup=time.monotonic(); train,valid,origin=data.load_train_valid('wikics',bound(source_pins['train']),bound(source_pins['development']))
        ordered={'x':train['x'],'edge_index':train['edge_index'],'train_ids':train['ids'],'train_y':train['y'],'valid_ids':valid['ids'],'valid_y':valid['y']}
        for name,tensor in ordered.items():
            import hashlib
            require(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()==source_pins['ordered_raw_tensor_fingerprints'][name]['contiguous_raw_bytes_sha256'],'Original ordered raw role fingerprint: '+name)
        iterator=data.validation_batches('wikics',train,valid,'cuda:0'); batch,truth=next(iterator)
        require(next(iterator,None) is None and len(truth)==5274 and tuple(batch['x'].shape)==(11701,300) and tuple(batch['edge_index'].shape)==(2,442907),'Complete original fullgraph development batch')
        torch.cuda.synchronize(); cost['data_loading_and_transfer_seconds']=time.monotonic()-setup; cost['data']=origin
        baseline=[row for seed in SEEDS for row in records if row['seed']==seed and row['condition']=='plain']; remaining=[row for row in records if row['condition']!='plain']
        for stage,cells in (('plain',baseline),('remaining',remaining)):
            if stage=='remaining':
                write(output/'compact'/'BASELINE_COHORTS.json',dict(cohorts=collection['cohorts'],frozen_before_candidate_prediction_inspection=True,definition='Same-seed plain only; overlapping fixed Wiki24 cohorts'))
            for record in cells:
                cohort=collect_cell(torch,np,public,adapter,analysis,contract,sources,batch,truth,record,source_pins,output,cost)
                if record['condition']=='plain': collection['cohorts'][str(record['seed'])]=cohort or dict(available=False,failure=record.get('failure'))
                write(output/'compact'/'COLLECTION.json',collection); write(output/'compact'/'COST.json',cost)
        collection['status']='complete' if all(row['collection_status']=='complete' for row in records) else 'complete_with_retained_collection_failures'
        write(output/'compact'/'COLLECTION.json',collection); setup=time.monotonic(); analysis.run(output,collection,pins)
        cost['analysis_seconds']=time.monotonic()-setup; cost['status']=collection['status']
    except Exception as error:
        collection['status']='retained_pipeline_failure'; collection['failure']=dict(error_type=type(error).__name__,error=str(error),automatic_retry=False)
        write(output/'compact'/'FAILURE.json',collection['failure']); write(output/'compact'/'COLLECTION.json',collection); raise
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic()-started,CPU_user_seconds=usage.ru_utime-usage_start.ru_utime,CPU_system_seconds=usage.ru_stime-usage_start.ru_stime,
            peak_RSS_bytes=int(usage.ru_maxrss*(1024 if sys.platform.startswith('linux') else 1)),raw_server_only_storage_bytes=sum(f.stat().st_size for f in (output/'raw').iterdir() if f.is_file()),
            per_cell_costs=[{k:v for k,v in row.items() if k in ('cell','seed','condition','collection_status') or k.endswith('_seconds') or k.startswith('peak_') or k.endswith('member_forwards')} for row in records],finished_UTC=datetime.now(timezone.utc).isoformat())
        write(output/'compact'/'COST.json',cost)
    print(str(output/'compact'))


if __name__=='__main__': main()
