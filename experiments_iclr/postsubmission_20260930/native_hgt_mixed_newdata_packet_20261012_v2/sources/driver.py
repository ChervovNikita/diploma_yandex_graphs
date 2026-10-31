"""Prepared HGB-DBLP driver. Root freeze/release required; test labels stay closed."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parent
SEEDS = (131,137,139,149,151)
ARMS = ('native_HGT','global_BE','CP','unrestricted','shared_relation','untied_HGT','wider_BE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def verified(record):
    path = Path(record['path']); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    require(digest.hexdigest()==record['sha256'] and ('bytes' not in record or path.stat().st_size==record['bytes']),
            'Packet fingerprint mismatch: '+str(path))
    return path


class NativeEarlyStop:
    """Exact author delta0: ties replace checkpoint and reset patience."""
    def __init__(self):
        self.best = None; self.counter = 0

    def observe(self, loss):
        require(math.isfinite(loss),'Nonfinite validation CE')
        if self.best is None or loss<=self.best:
            self.best = loss; self.counter = 0; return True,False
        self.counter += 1; return False,self.counter>=30


def freeze_guard(frozen, release, frozen_sha, manifest_sha):
    require(frozen['dataset']=='HGB-DBLP' and frozen['scope']=='complete_release_development_only',
            'Complete DBLP development scope required')
    require(frozen['seeds']==list(SEEDS),'Previously proposed paired seeds differ; root must resolve use history')
    require(frozen['arms'] and len(frozen['arms'])==len(set(frozen['arms'])) and set(frozen['arms'])<=set(ARMS),
            'Explicit supported arm freeze required')
    require({'CP','global_BE','native_HGT','shared_relation','unrestricted'}<=set(frozen['arms']),
            'Representative interpretation requires native/global/shared-relation/CP/unrestricted controls')
    require(frozen['recipe']==dict(width=64,heads=8,layers=3,use_norm=True,feature_type=2,
            dropout=0.2,max_epochs=300,patience=30,weight_decay=0.0001,
            optimizer='AdamW constructor default lr/betas/eps',scheduler='OneCycleLR max_lr=1e-3 total_steps=300 pct_start=0.05; step(epoch+1)',
            selection='post-update validation mean-logit CE; latest tied minimum; native delta0'),
            'Pinned native recipe differs')
    require(frozen['member_stream_rule']=='seed+700001+1009*member; persistent independent streams',
            'Prospective paired member RNG policy differs')
    require(frozen['factor_seed_rule']=='seed+900001' and frozen['untied_init_rule']=='member0 paired core; member>0 seed+200003*member',
            'Prospective paired initialization policy differs')
    require(frozen['shared_relation_control']=='optional unadopted until listed in frozen arms; rho0=0.01, same q/u/base signs; train rho/q/u',
            'Shared-control initialization declaration differs')
    require(frozen['test_labels_closed'] is True and frozen['study_adopted_by_root'] is True,
            'Root must settle study adoption before execution')
    require(release['execution_authorized'] is True and release['study_freeze_sha256']==frozen_sha
            and release['prepared_manifest_sha256']==manifest_sha and release['CPU_equivalence_passed'] is True
            and release['training_packet_CPU_fixture_passed'] is True
            and release['implementation_manifest_sha256']==frozen['implementation_manifest_sha256'],
            'Independent root source/CPU qualification and exact freeze release required')
    require(frozen['archive']['sha256']=='0d3ea4a74399f9cd3e83af206e8e0b67e1844fe2c8463b424189884dd58ad7c8'
            and frozen['archive']['bytes']==2567741,'Exact acquired DBLP archive required')
    require([(row['seed']) for row in frozen['splits']]==list(SEEDS),'All frozen paired splits required')


def paired_uncertainty(values):
    """Descriptive five-seed uncertainty, not power or dataset-generalization."""
    require(len(values)==5 and all(math.isfinite(x) for x in values),'All five finite paired deltas required')
    mean = sum(values)/5
    sd = math.sqrt(sum((x-mean)**2 for x in values)/4); se = sd/math.sqrt(5)
    return dict(mean=mean,paired_seed_SD=sd,paired_seed_SE=se,
                illustrative_t95_interval=[mean-2.7764451051977987*se,mean+2.7764451051977987*se],
                assumptions='five paired seed deltas; t interval assumes independent approximately normal deltas; not independent datasets or power')


def metrics(torch, member_logits, ids, labels, classes):
    logits = member_logits.mean(0)[ids]
    nll = float(torch.nn.functional.cross_entropy(logits,labels))
    predictions = logits.argmax(-1)
    accuracy = float((predictions==labels).double().mean())
    f1 = []
    for cls in range(classes):
        tp = int(((predictions==cls)&(labels==cls)).sum())
        fp = int(((predictions==cls)&(labels!=cls)).sum()); fn = int(((predictions!=cls)&(labels==cls)).sum())
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return dict(validation_NLL=nll,validation_micro_F1=accuracy,validation_macro_F1=sum(f1)/classes,
                split='development validation',calibrated=False,heldout=False)


def fit(torch, implementation, model, graph, features, split, train_labels, validation_labels, seed, arm, out, bindings, onecycle):
    out.mkdir()
    device = next(model.parameters()).device
    torch.manual_seed(seed+700001)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed+700001); torch.cuda.reset_peak_memory_stats(device)
    streams = implementation.MemberStreams([seed+700001+1009*m for m in range(4)],device)
    ids = torch.tensor(split['train_ids'],dtype=torch.long,device=device)
    val_ids = torch.tensor(split['validation_ids'],dtype=torch.long,device=device)
    labels = torch.tensor(train_labels,dtype=torch.long,device=device)
    val_labels = torch.tensor(validation_labels,dtype=torch.long,device=device)
    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=300,max_lr=1e-3,pct_start=.05)
    stopper = NativeEarlyStop(); selected = None; started = time.perf_counter()
    for epoch in range(300):
        model.train(); optimizer.zero_grad()
        logits = model(graph,features,'0',streams)
        train_loss = implementation.mean_member_ce(logits,ids,labels)
        require(bool(torch.isfinite(train_loss)),'Nonfinite TRAIN mean-member CE')
        train_loss.backward(); optimizer.step(); scheduler.step(epoch+1)
        model.eval()
        with torch.no_grad():
            member_logits = model(graph,features,'0')
            score = metrics(torch,member_logits,val_ids,val_labels,model.core.out.out_features if hasattr(model,'core') else model.cores[0].out.out_features)
        save,stop = stopper.observe(score['validation_NLL'])
        row = dict(epoch=epoch+1,train_mean_member_CE=float(train_loss.detach()),**score,
                   learning_rate=[group['lr'] for group in optimizer.param_groups],patience_counter=stopper.counter,checkpoint_replaced=save)
        with (out/'TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if save:
            selected = dict(epoch=epoch+1,**score)
            # State tensors are saved immediately; no later optimizer update can
            # mutate a retained best-state reference. No label payload saved.
            checkpoint = dict(schema='DBLP_selected_complete_state_v1',arm=arm,seed=seed,selection=selected,
                model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=onecycle.portable_state(scheduler),
                optimizer_parameter_names=[name for name,_ in model.named_parameters()],
                member_RNG=streams.state_dict(),torch_CPU_RNG=torch.get_rng_state(),
                torch_device_RNG=None if device.type=='cpu' else torch.cuda.get_rng_state(device),
                early_stopping=dict(best=stopper.best,counter=stopper.counter),bindings=bindings)
            torch.save(checkpoint,out/'selected.pt')
            torch.save(member_logits.detach().cpu(),out/'selected_member_logits.pt')
        if stop:
            break
    require(selected is not None,'No eligible post-update checkpoint')
    saved = torch.load(out/'selected.pt',map_location=device,weights_only=True)
    require(saved['optimizer_parameter_names']==[name for name,_ in model.named_parameters()],
            'Selected optimizer parameter-name order differs')
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); onecycle.restore_state(scheduler,saved['scheduler'])
    streams.load_state_dict(saved['member_RNG']); torch.set_rng_state(saved['torch_CPU_RNG'].cpu())
    if device.type=='cuda':
        torch.cuda.set_rng_state(saved['torch_device_RNG'].cpu(),device)
    model.eval()
    with torch.no_grad():
        replay = model(graph,features,'0')
        score = metrics(torch,replay,val_ids,val_labels,len(set(train_labels)))
    require(abs(score['validation_NLL']-selected['validation_NLL'])<=1e-7,'Selected-state validation NLL replay mismatch')
    result = dict(status='selected',seed=seed,arm=arm,selection=selected,updates=epoch+1,
         parameters=implementation.parameter_count(model),wall_seconds=time.perf_counter()-started,
         GPU_peak_allocated=None if device.type=='cpu' else torch.cuda.max_memory_allocated(device),
         GPU_peak_reserved=None if device.type=='cpu' else torch.cuda.max_memory_reserved(device),
         final_labels_closed=True,label_payloads_saved=False,selected_state_replay=True)
    write(out/'SELECTION.json',result); return result


def paired_development(rows, arms):
    complete = len(rows)==len(SEEDS)*len(arms) and all(row['status']=='selected' for row in rows)
    if not complete:
        return dict(status='incomplete',all_frozen_terminals=len(rows)==len(SEEDS)*len(arms),successful_subset_scored=False)
    scores = {arm:[next(row['selection']['validation_NLL'] for row in rows if row['seed']==seed and row['arm']==arm)
                   for seed in SEEDS] for arm in arms}
    deltas = {arm:[x-y for x,y in zip(scores['CP'],values)] for arm,values in scores.items() if arm!='CP'}
    return dict(status='complete_development_summary',paired_seed_order=list(SEEDS),validation_NLL=scores,
                CP_minus_control_validation_NLL=deltas,mean_deltas={arm:sum(values)/len(values) for arm,values in deltas.items()},
                descriptive_paired_uncertainty={arm:paired_uncertainty(values) for arm,values in deltas.items()},
                uncertainty_status='future heldout paired seed uncertainty requires separately frozen final evaluation; no significance claim',
                heldout_opened=False,practical_two_graph_gate_evaluated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',required=True); parser.add_argument('--admission',required=True)
    parser.add_argument('--run-name',required=True); parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args(argv)
    require(args.run_name not in ('','.', '..') and Path(args.run_name).name==args.run_name,'Fresh simple run name required')
    manifest_bytes = (PACKET/'MANIFEST.json').read_bytes(); manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    require(json.loads((PACKET/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Packet seal mismatch')
    for row in json.loads(manifest_bytes)['payload']:
        verified(dict(row,path=str(PACKET/row['path'])))
    provenance = json.loads((PACKET/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        verified(dict(row,path=str(PACKET.parent/row['path'])))
    inputs = module('prepared_DBLP_inputs',PACKET/'dblp_inputs.py')
    frozen_bytes = Path(args.freeze).read_bytes(); frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    frozen = json.loads(frozen_bytes); release_bytes = Path(args.admission).read_bytes(); release = json.loads(release_bytes)
    freeze_guard(frozen,release,frozen_sha,manifest_sha)
    require(release['run_name']==args.run_name and release['device']==args.device,'Root run/device release differs')
    require(frozen['implementation_manifest_sha256']==provenance['implementation_manifest_sha256'],'Qualified implementation changed')
    require(not list((PACKET/'runs').glob('*/STUDY_STARTED.json')),'No automatic restart/replacement study')
    out = PACKET/'runs'/args.run_name; out.mkdir(parents=True,exist_ok=False)
    admission = dict(study_freeze_sha256=frozen_sha,prepared_manifest_sha256=manifest_sha,
                     root_release_sha256=hashlib.sha256(release_bytes).hexdigest(),test_labels_closed=True)
    write(out/'STUDY_STARTED.json',admission)
    rows = []; summary = None; before = []; study_started = time.perf_counter()
    preservation_error = None
    try:
        before = [frozen['archive'],frozen['development_labels']]+[row['descriptor'] for row in frozen['splits']]
        for row in before:
            inputs.verified(row)
        schema,attributes,edges = inputs.stream_schema(frozen['archive'],frozen['members'])
        require(schema['member_sha256']==frozen['member_sha256'],'Byte-bound node/link payload differs')
        require(schema['node_counts']==frozen['expected_node_counts'],'Full released type counts differ')
        require(schema['input_dims']==frozen['expected_input_dims'],'Native feature2 dimensions differ')
        require({str(row['raw_id']):(row['source'],row['target']) for row in schema['relations']}==
                {key:tuple(value) for key,value in frozen['expected_relations'].items()},'Raw directed relation scope differs')
        development = inputs.read_development_labels(frozen['development_labels'],frozen['archive']['sha256'])
        require(development['source_label_member_sha256']==frozen['source_label_member_sha256'],
                'Explicit development input must bind original label.dat bytes')
        require(max(development['node_ids'])<schema['node_counts']['0'],'Development ID outside target type')
        # All release/source/root admission checks precede any native runtime import.
        import torch
        require(torch.__version__.split('+')[0]=='2.1.2','Pinned Torch runtime differs')
        device = torch.device(args.device)
        require(device.type in ('cpu','cuda'),'Pinned CPU/CUDA backend only')
        if device.type=='cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES')==frozen['GPU_uuid'],
                    'Root must expose exactly the prospectively bound GPU UUID')
            require(torch.cuda.device_count()==1 and device.index in (None,0),'One visible GPU required')
            device = torch.device('cuda:0')
        implementation_path = PACKET.parent/provenance['implementation_source']
        implementation = module('qualified_HGB_HGT_private',implementation_path)
        families = module('prepared_DBLP_families',PACKET/'families.py')
        onecycle = module('prepared_DBLP_portable_OneCycle',PACKET/'onecycle_state.py')
        graph,features,schema = inputs.materialize(schema,attributes,edges,implementation,frozen['relation_row_order'],device)
        write(out/'GRAPH_SCHEMA.json',schema); del attributes,edges
        bindings = dict(admission,archive=frozen['archive'],development_labels=frozen['development_labels'],
                        implementation_manifest_sha256=frozen['implementation_manifest_sha256'],graph_schema=schema)
        for split_record in frozen['splits']:
            seed = split_record['seed']
            split,train_labels,val_labels = inputs.verify_split(split_record['descriptor'],development,seed)
            models = families.build(implementation,graph,schema['input_dims'],len(development['train_class_schema']),
                                    seed,seed+900001,device,frozen['arms'])
            for arm in frozen['arms']:
                case = out/f'seed{seed}'/arm; case.parent.mkdir(exist_ok=True); arm_started = time.perf_counter()
                try:
                    rows.append(fit(torch,implementation,models[arm],graph,features,split,train_labels,val_labels,
                                    seed,arm,case,dict(bindings,split=split_record),onecycle))
                except Exception as error:
                    case.mkdir(exist_ok=True)
                    status = 'resource_deferred' if isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)) else 'failed'
                    row = dict(status=status,seed=seed,arm=arm,error_type=type(error).__name__,error_message=str(error),paid_wall_seconds=time.perf_counter()-arm_started,
                               traceback=traceback.format_exc(),final_labels_closed=True,successful_subset_scored=False)
                    write(case/'FAILURE.json',row); rows.append(row)
                finally:
                    del models[arm]
        summary = paired_development(rows,frozen['arms'])
    except Exception as error:
        error_record = dict(error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
        write(out/'STUDY_FAILURE.json',error_record)
        for seed in SEEDS:
            for arm in frozen['arms']:
                if not any(row['seed']==seed and row['arm']==arm for row in rows):
                    rows.append(dict(seed=seed,arm=arm,status='blocked',attempted=False,reason='study prerequisite failed',**error_record))
        summary = paired_development(rows,frozen['arms'])
    finally:
        try:
            for row in before:
                inputs.verified(row)
            for row in provenance['inputs']:
                inputs.verified(dict(row,path=str(PACKET.parent/row['path'])))
            for row in json.loads(manifest_bytes)['payload']:
                inputs.verified(dict(row,path=str(PACKET/row['path'])))
            require(hashlib.sha256(Path(args.freeze).read_bytes()).hexdigest()==frozen_sha
                    and Path(args.admission).read_bytes()==release_bytes,'Root freeze/release changed')
        except Exception as error:
            preservation_error = str(error); summary = None
        write(out/'STUDY.json',dict(schema='HGB_DBLP_development_training_v1',rows=rows,
              summary=summary,wall_seconds=time.perf_counter()-study_started,
              original_inputs_verified_unchanged=preservation_error is None,preservation_error=preservation_error,
              final_labels_closed=True,admission=admission,comparison_scope='HGT family only; competent native challengers remain separate required work'))
    print(json.dumps(dict(output=str(out),summary=summary)))
    return 0 if summary is not None and summary['status']=='complete_development_summary' else 1


if __name__=='__main__':
    raise SystemExit(main())
