"""Prepared HGB-ACM native15; root source/resource admission required; TEST closed."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parent
SEEDS = [131,137,139,149,151]
ARMS = ['native_GAT','native_Simple_HGN','native_SeHGNN']


def require(condition,message):
    if not condition:
        raise ValueError(message)


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def verify(record,path=None):
    path = Path(path or record['path']); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    require(digest.hexdigest()==record['sha256'] and path.stat().st_size==record['bytes'],
            'Fingerprint differs: '+str(path))
    return path


def bind_reused_functions(provenance):
    """Import the unchanged qualified stopping and RNG functions after hash checks."""
    existing = module('immutable_native_driver_for_ACM',PACKET.parent/provenance['training_source'])
    globals().update(NativeStopper=existing.NativeStopper,seed_all=existing.seed_all,
                     global_state=existing.global_state,restore_global=existing.restore_global)
    return existing



def validation_metrics(torch,logits,labels):
    predictions = logits.argmax(-1)
    values = []
    for c in range(3):
        tp = int(((predictions==c)&(labels==c)).sum())
        fp = int(((predictions==c)&(labels!=c)).sum()); fn = int(((predictions!=c)&(labels==c)).sum())
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(validation_NLL=float(torch.nn.functional.cross_entropy(logits,labels)),
                validation_micro_F1=float((predictions==labels).double().mean()),
                validation_macro_F1=sum(values)/3,heldout=False,calibrated=False)


def train_epoch(torch,model,optimizer,arm,graph,features,label_features,ids,labels,loader=None,scaler=None):
    model.train()
    if arm!='native_SeHGNN':
        optimizer.zero_grad(); logits = model(graph,features)[:len(features[0])]
        loss = torch.nn.functional.cross_entropy(logits[ids],labels)
        require(bool(torch.isfinite(loss)),'Nonfinite TRAIN CE')
        loss.backward(); optimizer.step(); return float(loss.detach())
    device = next(model.parameters()).device; losses = []
    lookup = {int(i):int(y) for i,y in zip(ids.cpu().tolist(),labels.cpu().tolist())}
    for batch in loader:
        f = {k:v[batch].to(device) for k,v in features.items()}
        l = {k:v[batch].to(device) for k,v in label_features.items()}
        y = torch.tensor([lookup[int(i)] for i in batch],dtype=torch.long,device=device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(batch,f,l); loss = torch.nn.functional.cross_entropy(logits,y)
            require(bool(torch.isfinite(loss)),'Nonfinite AMP TRAIN CE')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(batch,f,l); loss = torch.nn.functional.cross_entropy(logits,y)
            require(bool(torch.isfinite(loss)),'Nonfinite TRAIN CE')
            loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    require(bool(losses),'Empty native minibatch training')
    return sum(losses)/len(losses)


def predict_validation(torch,model,arm,graph,features,label_features,split,eval_ids):
    model.eval(); device = next(model.parameters()).device
    with torch.no_grad():
        if arm!='native_SeHGNN':
            ids = torch.tensor(split['validation_ids'],dtype=torch.long,device=device)
            return model(graph,features)[ids]
        # No inference autocast in native SeHGNN, even when TRAIN uses AMP.
        batch = torch.tensor(eval_ids,dtype=torch.long)
        output = model(batch,{k:v[batch].to(device) for k,v in features.items()},
                       {k:v[batch].to(device) for k,v in label_features.items()})
        begin = len(split['train_ids']); end = begin+len(split['validation_ids'])
        return output[begin:end]


def fit(torch,model,arm,graph,features,label_features,split,train_labels,val_labels,seed,out,bindings,eval_ids):
    out.mkdir(); device = next(model.parameters()).device
    ids = torch.tensor(split['train_ids'],dtype=torch.long,device=device)
    labels = torch.tensor(train_labels,dtype=torch.long,device=device)
    validation_labels = torch.tensor(val_labels,dtype=torch.long,device=device)
    sehgnn = arm=='native_SeHGNN'; lr,weight_decay = (.001,0.) if sehgnn else (.0005,.0001)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    loader = torch.utils.data.DataLoader(split['train_ids'],batch_size=10000,shuffle=True,drop_last=False) if sehgnn else None
    scaler = torch.cuda.amp.GradScaler() if sehgnn and device.type=='cuda' else None
    stopper = NativeStopper(arm); selected = None; started = time.perf_counter()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,(200 if sehgnn else 300)+1):
        loss = train_epoch(torch,model,optimizer,arm,graph,features,label_features,ids,labels,loader,scaler)
        logits = predict_validation(torch,model,arm,graph,features,label_features,split,eval_ids)
        score = validation_metrics(torch,logits,validation_labels)
        save,stop = stopper.observe(epoch,score['validation_NLL'])
        with (out/'TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(epoch=epoch,TRAIN_CE=loss,**score,checkpoint_replaced=save,
                native_patience_counter=stopper.counter),allow_nan=False)+'\n')
        if save:
            selected = dict(epoch=epoch,**score)
            checkpoint = dict(schema='ACM_native_challenger_complete_state_v1',arm=arm,seed=seed,
                model=model.state_dict(),optimizer=optimizer.state_dict(),AMP=None if scaler is None else scaler.state_dict(),
                global_RNG=global_state(torch,device),selection=selected,
                optimizer_parameter_names=[name for name,_ in model.named_parameters()],
                early_stopping=dict(best=stopper.best,best_epoch=stopper.best_epoch,counter=stopper.counter),
                fixed_evaluation_ids=eval_ids if sehgnn else None,bindings=bindings)
            torch.save(checkpoint,out/'selected.pt')
            torch.save(logits.detach().cpu(),out/'selected_validation_logits.pt')
        if stop:
            break
    require(selected is not None,'No eligible post-update checkpoint')
    saved = torch.load(out/'selected.pt',map_location=device,weights_only=True)
    require(saved['optimizer_parameter_names']==[name for name,_ in model.named_parameters()],
            'Optimizer parameter order changed')
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(saved['AMP'])
    restore_global(torch,saved['global_RNG'],device)
    replay = predict_validation(torch,model,arm,graph,features,label_features,split,eval_ids)
    score = validation_metrics(torch,replay,validation_labels)
    torch.testing.assert_close(replay.cpu(),torch.load(out/'selected_validation_logits.pt',weights_only=True),atol=1e-7,rtol=1e-7)
    require(abs(score['validation_NLL']-selected['validation_NLL'])<=1e-7,'Selected validation CE replay mismatch')
    result = dict(status='selected',arm=arm,seed=seed,selection=selected,epochs_paid=epoch,
        parameters=sum(p.numel() for p in model.parameters()),wall_seconds=time.perf_counter()-started,
        GPU_peak_allocated=None if device.type=='cpu' else torch.cuda.max_memory_allocated(device),
        GPU_peak_reserved=None if device.type=='cpu' else torch.cuda.max_memory_reserved(device),
        native_AMP_used=scaler is not None,TEST_label_reads=0,TEST_diagnostics=0,
        selected_state_replay=True,classifier_classes_from_TRAIN=3)
    write(out/'SELECTION.json',result); return result


def guard(frozen,release,freeze_sha,manifest_sha,provenance):
    require(frozen['dataset']=='HGB-ACM' and frozen['scope']=='complete_release_development_only',
            'Complete ACM native development scope required')
    require(frozen['seeds']==SEEDS and frozen['arms']==ARMS,'All15 paired native ACM terminals required')
    for key,expected in provenance['confirmation_bindings'].items():
        require(frozen[key]==expected,'Pinned ACM native binding differs: '+key)
    require(frozen['study_adopted_by_root'] is True and frozen['test_labels_closed'] is True,
            'Root adoption and closed TEST required')
    require(release['execution_authorized'] is True and release['study_freeze_sha256']==freeze_sha
            and release['prepared_manifest_sha256']==manifest_sha
            and release['scientific_design_sha256']==frozen['scientific_design_sha256']
            and release['DBLP_continuation_gate_passed'] is True
            and release['ACM_native_specific_check_passed'] is True,
            'Exact root release, DBLP continuation gate and ACM-specific preprocessing/resource/replay check required')
    paired = json.loads(verify(frozen['paired_HGT_freeze']).read_text())
    require(paired['dataset']==frozen['dataset'] and paired['scope']==frozen['scope']
            and paired['study_adopted_by_root'] is True and paired['test_labels_closed'] is True,
            'Adopted ACM paired HGT freeze with TEST closed required')
    for key in ('seeds','archive','development_labels','splits','source_label_member_sha256','scientific_design_sha256'):
        require(paired[key]==frozen[key],'Native/HGT binding differs: '+key)
    require(release['paired_HGT_freeze_sha256']==frozen['paired_HGT_freeze']['sha256'],
            'Release must bind exact paired HGT freeze')
    require(isinstance(frozen['GPU_uuid'],str) and frozen['GPU_uuid'].startswith('GPU-'),
            'Root one-GPU identity binding required')



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',required=True); parser.add_argument('--admission',required=True)
    parser.add_argument('--run-name',required=True); parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args(argv)
    require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.','..'),'Fresh simple run name required')
    manifest_bytes = (PACKET/'MANIFEST.json').read_bytes(); manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    payload = json.loads(manifest_bytes)['payload']; provenance = json.loads((PACKET/'PROVENANCE.json').read_text())
    for row in payload:
        verify(row,PACKET/row['path'])
    for row in provenance['inputs']:
        verify(row,PACKET.parent/row['path'])
    frozen_bytes,release_bytes = Path(args.freeze).read_bytes(),Path(args.admission).read_bytes()
    frozen,release = json.loads(frozen_bytes),json.loads(release_bytes)
    freeze_sha = hashlib.sha256(frozen_bytes).hexdigest()
    guard(frozen,release,freeze_sha,manifest_sha,provenance)
    require(release['run_name']==args.run_name and release['device']==args.device,'Root run/device release differs')
    require(not list((PACKET/'runs').glob('*/STUDY_STARTED.json')),'No automatic replacement/restart study')
    bind_reused_functions(provenance)
    loader = module('prepared_ACM_stream_loader_for_native',PACKET.parent/provenance['loader_source'])
    data = module('prepared_ACM_native_inputs',PACKET/'native_inputs.py')
    base_inputs = module('immutable_native_graph_helpers_for_ACM',PACKET.parent/provenance['native_inputs_source'])
    descriptors = [frozen['archive'],frozen['development_labels'],frozen['paired_HGT_freeze']]+[x['descriptor'] for x in frozen['splits']]
    for row in descriptors:
        loader.verified(row)
    out = PACKET/'runs'/args.run_name; out.mkdir(parents=True,exist_ok=False)
    admission = dict(study_freeze_sha256=freeze_sha,prepared_manifest_sha256=manifest_sha,
        scientific_design_sha256=frozen['scientific_design_sha256'],
        root_admission_sha256=hashlib.sha256(release_bytes).hexdigest(),TEST_labels_closed=True)
    write(out/'STUDY_STARTED.json',admission); rows = []; started = time.perf_counter(); preservation_error = None
    try:
        schema,attributes,edges = data.stream_all_attributes(loader,frozen['archive'],frozen['members'])
        require(schema['node_counts']==frozen['expected_node_counts']
                and schema['provided_attribute_widths']==frozen['expected_feature_widths'],
                'Complete ACM node/feature schema differs')
        require(schema['member_sha256']==frozen['member_sha256'],'Byte-bound ACM node/link members differ')
        statistics = {str(r['raw_id']):{k:r[k] for k in ('raw_records','support_edges','raw_self_records','duplicates_coalesced')}
                      for r in schema['relations']}
        require(statistics==frozen['expected_relation_statistics'],'Released ACM relation/self support differs')
        development = loader.read_development_labels(frozen['development_labels'],frozen['archive']['sha256'])
        require(development['source_member']=='ACM/label.dat'
                and development['source_member_sha256']==frozen['source_label_member_sha256']
                and development['train_class_schema']==[0,1,2] and len(development['node_ids'])==907,
                'Exact canonical ACM source development pool/classes required')
        require(max(development['node_ids'])<schema['node_counts']['0'],'Development target ID out of range')
        records = data.homogeneous_records(schema,edges)
        # Source/admission and input checks precede the numerical runtime import.
        import torch
        require(torch.__version__.split('+')[0]=='2.1.2','Pinned Torch2.1.2 required')
        device = torch.device(args.device)
        require(device.type in ('cpu','cuda'),'CPU/CUDA only')
        if device.type=='cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES')==frozen['GPU_uuid'] and torch.cuda.device_count()==1
                    and device.index in (None,0),'Root must expose the one bound GPU UUID')
            device = torch.device('cuda:0')
        models = module('immutable_native_models_for_ACM',PACKET.parent/provenance['models_source'])
        graph = base_inputs.materialize_homogeneous(records,models,device)
        gat_features = data.homogeneous_features(schema,attributes,device)
        require([x.shape[1] for x in gat_features]==[1902,5959,56,1902],
                'Native ACM feats2 dimensions differ')
        adjs,preprocessing_audit = data.normalized_native_adjacencies(schema,attributes,edges)
        features = data.feature_channels(schema,attributes,adjs); products = data.label_products(adjs)
        require(list(features)==frozen['feature_channel_order'] and list(products)==frozen['label_product_order'],
                'Frozen ACM native channel traversal differs')
        del attributes,edges,adjs
        schema.update(dataset='HGB-ACM',homogeneous_audit=records['audit'],SeHGNN_preprocessing_audit=preprocessing_audit,
            native_GAT_Simple_HGN_feature_type=2,native_GAT_Simple_HGN_input_dims=[1902,5959,56,1902],
            SeHGNN_feature_channels={k:list(v.shape) for k,v in features.items()},
            SeHGNN_label_channels=sorted(products),TEST_label_reads=0)
        write(out/'GRAPH_SCHEMA.json',schema)
        bindings = dict(admission,archive=frozen['archive'],development_labels=frozen['development_labels'],graph_schema=schema)
        for split_record in frozen['splits']:
            seed = split_record['seed']; split,train_labels,val_labels = loader.verify_split(split_record['descriptor'],development,seed)
            require(split['development_descriptor']==frozen['development_labels'],
                    'Split canonical development descriptor differs')
            require(len(train_labels)==726 and len(val_labels)==181,'Complete ACM80/20 development split required')
            label_features = data.train_only_label_channels(products,schema['node_counts']['0'],split['train_ids'],train_labels)
            eval_ids = base_inputs.native_evaluation_ids(schema['node_counts']['0'],split)
            receipt_path = out/f'seed{seed}'; receipt_path.mkdir(exist_ok=True)
            cache_receipt = data.channel_receipt(features,label_features,eval_ids)
            write(receipt_path/'PREPROCESSING.json',cache_receipt)
            for arm in ARMS:
                case = receipt_path/arm; paid = time.perf_counter(); model = None
                try:
                    seed_all(torch,seed,device)
                    if arm=='native_SeHGNN':
                        model = models.SeHGNN({k:v.shape[1] for k,v in features.items()},label_features.keys(),
                            classes=3,n_task_layers=1,residual=False).to(device)
                        model.tgt_type = 'P'
                        arm_features = features
                    else:
                        model = models.HGBGAT([1902,5959,56,1902],classes=3,
                            simple=arm=='native_Simple_HGN',num_etypes=17).to(device)
                        arm_features = gat_features
                    rows.append(fit(torch,model,arm,graph,arm_features,label_features,split,train_labels,val_labels,
                        seed,case,dict(bindings,split=split_record,preprocessing=cache_receipt),eval_ids))
                except Exception as error:
                    case.mkdir(exist_ok=True)
                    status = 'resource_deferred' if isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)) else 'failed'
                    row = dict(status=status,seed=seed,arm=arm,error_type=type(error).__name__,error_message=str(error),
                        traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-paid,TEST_label_reads=0,
                        successful_subset_scored=False)
                    write(case/'FAILURE.json',row); rows.append(row)
                finally:
                    del model
    except Exception as error:
        record = dict(error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
        write(out/'STUDY_FAILURE.json',record)
        for seed in SEEDS:
            for arm in ARMS:
                if not any(x['seed']==seed and x['arm']==arm for x in rows):
                    rows.append(dict(status='blocked',seed=seed,arm=arm,attempted=False,**record))
    finally:
        try:
            for row in descriptors:
                loader.verified(row)
            for row in payload:
                verify(row,PACKET/row['path'])
            for row in provenance['inputs']:
                verify(row,PACKET.parent/row['path'])
            require(Path(args.freeze).read_bytes()==frozen_bytes and Path(args.admission).read_bytes()==release_bytes,
                    'Root freeze/admission changed')
        except Exception as error:
            preservation_error = str(error)
        complete = len(rows)==15 and all(x['status']=='selected' for x in rows) and preservation_error is None
        summary = dict(status='complete_development_challengers' if complete else 'incomplete',all_frozen_terminals=len(rows)==15,
            successful_subset_scored=False,TEST_label_reads=0,TEST_diagnostics=0,paired_seed_order=SEEDS)
        if complete:
            summary['validation_NLL'] = {arm:[next(x['selection']['validation_NLL'] for x in rows if x['arm']==arm and x['seed']==seed) for seed in SEEDS] for arm in ARMS}
        write(out/'STUDY.json',dict(schema='HGB_ACM_native_development_challengers_v1',rows=rows,summary=summary,admission=admission,
            wall_seconds=time.perf_counter()-started,original_inputs_unchanged=preservation_error is None,
            preservation_error=preservation_error,study_superiority_claim=False,heldout=False,calibrated=False))
    print(json.dumps(dict(output=str(out),summary=summary)))
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
