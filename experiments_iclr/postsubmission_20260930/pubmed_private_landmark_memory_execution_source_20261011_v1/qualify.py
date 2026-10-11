"""One finite full-input source/all-interface material check; TRAIN-only truth."""
from pathlib import Path
from learning import own_steps,serve,timed,update
from runtime import ARMS,SEEDS,Runtime,receipt,require,write

def poison(rt,model,optimizer,streams):
    with rt.t.no_grad():
        for parameter in model.parameters():parameter.add_(1)
        for state in optimizer.state.values():
            for value in state.values():
                if rt.t.is_tensor(value):value.add_(1)
    for group in optimizer.param_groups:group['lr']=.7
    for module in model.modules():module.training=not module.training
    if isinstance(streams,dict):
        for index,key in enumerate(streams):
            fresh=rt.fresh_stream(711+index);streams[key]=dict(cpu=fresh['cpu'],cuda=fresh['cuda'])
    else:streams[:]=[rt.fresh_stream(811+i) for i in range(len(streams))]
    rt.restore_rng(rt.fresh_stream(997))

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None
    result=dict(stage='qualify',complete=False,status='running',fits=0,retry=False,config=config['_config_receipt'],engineering_policy=config['engineering_policy'],containment_tolerance=config['containment_tolerance'],custody=custody,source_manifest=config['source_manifest'],prepared_custody=config['prepared_custody'],prepare_owner_end=config['prepare_owner_end'],SELECT_CAL_DEVELOPMENT_truth_read=False,TEST_read=False,source_updates=0,stage2_updates=0,records=[],automatic_fits_released=False)
    try:
        require(config.get('qualification_seed')==SEEDS[0],'One fixed full-input block9101 material check')
        rt=Runtime(config);roles=rt.roles(select=False);seed=SEEDS[0];ancestors=[];caches=[];source_logits=[];source_records=[]
        for body in range(4):
            session,audit=rt.fresh_source(seed,body);native=session.bodies[0];optimizer=session.optimizers[0];streams=session.streams.states
            require(not optimizer.state,'Fresh source optimizer0')
            meta=dict(qualification='source_checkpoint0_discarded',seed=seed,body=body,source_manifest=config['source_manifest'],prepared_custody=config['prepared_custody'])
            path=output/f'source_body{body}_checkpoint0.pt';digest=rt.save(path,native,optimizer,streams,meta)
            trained,clock=timed(rt,lambda:session.train_step(audit=False));result['source_updates']+=1
            require(session.counters['updates']==1 and all(rt.t.isfinite(p.grad).all().item() for p in native.parameters() if p.grad is not None),'Actual complete23-site source TRAIN update')
            own_steps(rt,optimizer,1)
            poison(rt,native,optimizer,streams);require(rt.restore(path,native,optimizer,streams,meta)==digest and not optimizer.state,'Full source model/Adam0/all modes/global/private RNG fresh restore')
            cache,original,cache_record=rt.cache_source(session,seed,body,receipt(path))
            ancestors.append(native);caches.append(cache);source_logits.append(original)
            source_records.append(dict(body=body,construction=audit,full_TRAIN=trained,TRAIN_timing=clock,checkpoint0=receipt(path),checkpoint0_fingerprint=digest,strict_full_source_restore=True,restored_Adam_parameter_states=0,cache=cache_record))
        result['source_records']=source_records;result['graph']=rt.graph
        paired_initializers={}
        for arm in ARMS:
            core,optimizers,audit=rt.construct(arm,seed,ancestors,caches);row=dict(arm=arm,construction=audit,bodies=[],I4_wholly_disjoint=core['genuine_independent']);result['records'].append(row)
            for body,(model,optimizer) in enumerate(zip(core['models'],optimizers)):
                streams=rt.streams(seed,body,model.views);require(not optimizer.state,'Fresh own stage2 Adam0')
                if model.memory is not None:
                    initial=rt.fp({name:module.weight for name,module in model.memory.maps.items() if name!='Wo'})
                    if body in paired_initializers:require(paired_initializers[body]==initial,'Paired six-map initial state across actual arms')
                    else:paired_initializers[body]=initial
                (logits,probability),clock=timed(rt,lambda:serve(rt,model));original=source_logits[body]
                require(all(rt.t.equal(z.max(1)[1],original.max(1)[1]) for z in logits) and all(rt.t.allclose(z,original,**config['containment_tolerance']) for z in logits),'Checkpoint0 complete source-classifier prediction/float containment')
                del logits,probability
                meta=dict(qualification='stage2_checkpoint0_discarded',seed=seed,arm=arm,body=body,source_manifest=config['source_manifest'],prepared_custody=config['prepared_custody'])
                path=output/(arm.replace('[','_').replace(']','')+f'_body{body}_checkpoint0.pt');digest=rt.save(path,model,optimizer,streams,meta)
                body_record=dict(body=body,checkpoint0=receipt(path),checkpoint0_fingerprint=digest,checkpoint0_full_containment=True,checkpoint0_serving_timing=clock,updates=[])
                for step in range(1,3 if model.memory is not None else 2):
                    actual,clock=timed(rt,lambda:update(rt,model,optimizer,streams,roles,gradient_receipt=True));own_steps(rt,optimizer,step);result['stage2_updates']+=1
                    body_record['updates'].append(dict(step=step,actual_TRAIN=actual,timing=clock))
                    if step==2:
                        memory_gradients={name:entry for name,entry in actual['gradients'].items() if name.startswith('memory.')}
                        require(memory_gradients and all(entry['nonzero'] for entry in memory_gradients.values()),'Every live memory map/E/FS64 r/s has nonzero finite CE gradient after readout learns')
                        if arm=='LM4':require(all(bool((model.memory.E.grad[v]!=0).any().item()) for v in range(4)),'Each actual private offset row receives nonzero CE gradient')
                        body_record['all_memory_CE_gradients_nonzero_step2']=True
                poison(rt,model,optimizer,streams);restored,restore_clock=timed(rt,lambda:rt.restore(path,model,optimizer,streams,meta))
                require(restored==digest and not optimizer.state,'Exact stage2 checkpoint0 restoration with empty moments')
                (logits,probability),serve_clock=timed(rt,lambda:serve(rt,model));require(all(rt.t.equal(z.max(1)[1],original.max(1)[1]) for z in logits),'Restored full source-contained native decisions')
                body_record.update(strict_model_optimizer_all_modes_global_and_private_RNG_restore=True,restored_Adam_parameter_states=0,restore_timing=restore_clock,restored_serving_timing=serve_clock)
                row['bodies'].append(body_record);del logits,probability,streams
            # Measure actual complete I4 serving with every independently
            # restored body resident. No best-member or pooled selector.
            if core['genuine_independent']:
                def assembled():
                    logits=rt.t.cat([serve(rt,m)[0] for m in core['models']],dim=0)
                    return logits,logits.softmax(-1).mean(0)
                (logits,probability),clock=timed(rt,assembled)
                require(logits.shape==(4,19717,3) and rt.t.isfinite(probability).all().item(),'Complete restored equipped four-body serving')
                row['full_independent_serving_timing']=clock;del logits,probability
            del core,optimizers,model,optimizer;rt.t.cuda.empty_cache();rt.guard()
        require(len(result['records'])==8 and result['source_updates']==4 and result['stage2_updates']==26,'One complete fixed source/interface material check inventory')
        result.update(complete=True,status='pass',interfaces=8,source_bodies=4,source_manifest=config['source_manifest'],core_source=receipt(Path(__file__).with_name('core.py')),learning_source=receipt(Path(__file__).with_name('learning.py')),checkpoint0_empty_moments_and_full_restore=True,all_memory_second_step_CE_gradients_verified=True,no_selected_ancestor_reuse=True,meaning='Complete full-input engineering evidence only; no new quality score or automatic fit admission')
    except BaseException as error:
        result.update(status='failed_retained',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if rt is not None:result['costs']=rt.costs()
        write(output/'QUALIFICATION.json',result)
    return result
