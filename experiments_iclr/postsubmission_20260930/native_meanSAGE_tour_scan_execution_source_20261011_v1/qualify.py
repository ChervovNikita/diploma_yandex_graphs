"""One full-input SW4 update/restore and label-free SELECT forward; no fit."""
import json
from pathlib import Path
from acquire import criterion
from runtime import ARMS,BLOCKS,Runtime,checked,receipt,require,write

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None
    result=dict(complete=False,status='running',fits=0,TRAIN_routes=0,Adam_steps=0,SELECT_label_free_routes=0,SELECT_labels_read=False,CAL_DEV_truth_read=False,custody=custody,role_custody=config['custody'],source_manifest=config['source_manifest'],automatic_fit_release=False)
    try:
        from reader import owner_closed
        owner_closed(config['derivative_owner_end'])
        derivative=json.loads(checked(config['derivative']).read_text())
        checked(config['derivative_source_manifest'])
        require(config['derivative_source_manifest']['sha256'] in (config['source_manifest']['sha256'],'041c1348cb7e730efd077e589be86069c46d74fa1dcd1ec9397fa204e61b1ae0'),'Derivative source must be this exact execution seal or its exact bound core seal')
        require(derivative['complete'] and derivative['fixtures']==1 and derivative['CPUfloat64'] and derivative['nonzero_output_projection'] and derivative['source_manifest']==config['derivative_source_manifest'],'The single independent nonzero derivative check closes first')
        rt=Runtime(config);roles=rt.roles(BLOCKS[0],select=False);result['graph']=rt.graph
        result['tour_bindings']=rt.tour_bindings;result['derivative']=config['derivative']
        inspections=[]
        for arm in ARMS:
            c,o,audit=rt.construct(arm,BLOCKS[0]);inspections.append(dict(arm=arm,**audit));del c,o
        result['all_frozen_arm_structural_inspections']=inspections
        core,optimizers,audit=rt.construct('SW4',BLOCKS[0]);model=core['models'][0];optimizer=optimizers[0];streams=rt.streams(BLOCKS[0],4)
        model.train();optimizer.zero_grad();out=rt.forward(model,streams);result['TRAIN_routes']=4
        loss,ce,penalty=criterion(rt,model,out,roles['train_ids'],roles['train_y']);require(rt.t.isfinite(loss).item(),'Finite full-TRAIN objective');loss.backward()
        require(all(p.grad is not None and rt.t.isfinite(p.grad).all().item() for p in model.parameters()),'All registered gradients finite, including zero-start scan maps')
        optimizer.step();result['Adam_steps']=1
        require(optimizer.state['tf1_global']['updates']==1 and all(optimizer.state[p] for p in model.parameters()),'Exactly one source Adam advancement and complete moments')
        del out,loss,ce,penalty;rt.guard();meta=dict(qualification='discarded_one_update',binding=custody['binding'],role_custody=config['custody'])
        path=output/'discarded_state.pt';saved=rt.save(path,model,optimizer,streams,meta)
        with rt.t.no_grad():
            for p in model.parameters():p.add_(1.)
            for values in optimizer.state.values():
                for v in values.values():
                    if rt.t.is_tensor(v):v.add_(1.)
            optimizer.state['tf1_global']['updates']+=7;optimizer.param_groups[0]['lr']=.7
        model.eval();streams[0]=rt.streams(BLOCKS[1],1)[0]
        rt.restore_rng(rt.streams(BLOCKS[1],1)[0]);rt.restore(path,model,optimizer,streams,meta)
        require(optimizer.state['tf1_global']['updates']==1 and model.training,'Exactly one restored Adam update and TRAIN mode')
        result['strict_model_optimizer_mode_global_and_route_rng_restore']=True;result['discarded_state']=receipt(path);result['saved_state_fingerprint']=saved
        model.eval()
        with rt.t.no_grad():
            out=rt.forward(model);selected=out[:,roles['select_ids']];probability=selected.softmax(-1).mean(0)
            require(selected.shape==(4,150,5) and rt.t.isfinite(probability).all().item(),'Full label-free SELECT output')
        result['SELECT_label_free_routes']=4;rt.guard();result.update(complete=True,status='pass',construction=audit,costs=rt.costs(),meaning='One actual full-input resource/restore check; no quality/novelty claim')
    except BaseException as e:
        result.update(status='failed_retained',failure=dict(type=type(e).__name__,message=str(e)),retry=False);raise
    finally:
        if rt is not None:result['costs']=rt.costs()
        write(output/'QUALIFICATION.json',result)
    return result
