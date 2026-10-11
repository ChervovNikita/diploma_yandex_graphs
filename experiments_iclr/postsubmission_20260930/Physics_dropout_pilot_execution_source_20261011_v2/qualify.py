"""One full candidate update and one active-four coupled I4 round; no selector."""
from pathlib import Path
import time
from runtime import Runtime,require,receipt,write

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,SELECT_array_read=False,CAL_DEV_truth_read=False,numerical_quality_claim=False,retry=False,records=[])
    try:
        rt=Runtime(config);t=rt.t;roles=rt.roles(8101,select=False)
        for coupled in (False,True):
            count=4 if coupled else 1;arm='unshared_coupled_I4' if coupled else 'O4_graph_input_sign_pair'
            models=[];optimizers=[];streams=[]
            for m in range(count):
                model,optimizer,stream=rt.construct(8101+1000003*m if coupled else 8101,1 if coupled else 4)
                models.append(model);optimizers.append(optimizer);streams.append(stream)
            generator=rt.generator(8101);sampler=rt.samplers.make_mask_sampler('activation_sign_paired',torch=t,generator=generator)
            draws={'calls':0,'finite':True}
            def checked_sampler(*args,**kwargs):
                value=sampler(*args,**kwargs);draws['calls']+=1
                require(t.isfinite(value).all().item() and not value.requires_grad,'Finite detached masks')
                return value
            sampler_before=rt.fp(generator.get_state());tick=time.monotonic()
            if coupled:
                update=rt.independent.coupled_I4_round(models,optimizers,[True]*4,rt.data,rt.data.x,roles['train_ids'],roles['train_y'],torch=t,functional=rt.F,mask_sampler=checked_sampler,native_route_forward=rt.joint.native_route_forward,body_rng_scope=lambda i:rt.body_scope(streams,i))
                losses=update['losses']
            else:
                with rt.body_scope(streams,0):
                    losses=[rt.joint.train_step_joint_routes(models[0],rt.data,rt.data.x,roles['train_ids'],roles['train_y'],optimizers[0],placement='first_normalized_sage_input',torch=t,functional=rt.F,mask_sampler=checked_sampler)]
                update=dict(own_updates=1,prefix_forwards=4,TRAIN_suffix_routes=4)
            require(all(__import__('math').isfinite(x) for x in losses),'Finite full-input own loss')
            require(draws['calls']==1 and sampler_before!=rt.fp(generator.get_state()),'Persistent sampler consumed one round')
            rt.guard();meta=dict(arm=arm,block=8101,step=1,custody=custody,role_custody=config['custody'])
            path=output/(arm+'.pt');digest=rt.save(path,models,optimizers,streams,generator,meta)
            # Deliberately change every state category before one strict restore.
            with t.no_grad():
                for model in models:
                    for p in model.parameters():p.add_(1)
                for optimizer in optimizers:
                    for state in optimizer.state.values():
                        for value in state.values():
                            if t.is_tensor(value):value.add_(1)
            rt.common.set_seed(17);streams[:]=[rt.rng() for _ in streams]
            t.rand((1,),device=rt.device,generator=generator)
            require(rt.fp([m.state_dict() for m in models])!=rt.fp(t.load(path,map_location='cpu',weights_only=True)['models']),'Parameters actually poisoned')
            rt.restore(path,models,optimizers,streams,generator,meta);rt.guard()
            result['records'].append(dict(arm=arm,full_graph=True,own_updates=update['own_updates'],prefix_forwards=4,TRAIN_suffix_routes=4,backward_calls=4,finite_loss=True,finite_masks=True,sampler_calls=1,sampler_persistent=True,strict_model_optimizer_global_body_sampler_restore=True,checkpoint=receipt(path),state_fingerprint=digest,seconds=time.monotonic()-tick,costs=rt.costs()))
            del models,optimizers,streams,generator,sampler,model,optimizer,stream
            t.cuda.empty_cache()
        result.update(complete=True,full_input_candidate_updates=1,full_input_coupled_rounds=1,total_optimizer_updates=5,total_backwards=8,scope='execution only; source-scoped sampler law',costs=rt.costs(),custody=custody,role_custody=config['custody'])
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e));write(output/'QUALIFICATION.json',result);raise
    write(output/'QUALIFICATION.json',result);return result
