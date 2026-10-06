#!/usr/bin/env python3
"""Disabled full-TRAIN numerical qualification; all references/states discarded."""
import copy
import time
from common import guard,runtime,load_data,write,sha,deadline,compare,TOLERANCES


def main():
    args,job,output=guard(__file__,'numerical');torch,versions=runtime()
    import numpy as np
    from method import (construct_native,native_hidden_logits,ResidualBank,PathSingle,correction_loss,
        bank_epoch,single_epoch,cpu_tree,rng_snapshot,rng_restore,adam,active_native,optimizer_ownership)
    data,authority=load_data(torch,job,True)
    x,edge,ids,labels=(data[k] for k in ('x','edge_index','train_ids','train_y'))
    output.mkdir();started=time.monotonic();old_rng=rng_snapshot()
    results={'kind':'numerical','source_manifest_sha256':job['source_manifest_sha256'],'runtime':versions,
        'TRAIN_only':True,'VALID_values_access':False,'TEST_access':False,'full_nodes':11701,'edges':442907,
        'TRAIN_labels':580,'full1100_fits':0,'states_discarded':True,'checks':[],'thresholds':TOLERANCES}

    def same_rng(a,b):
        return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])

    def unchanged(model,state):
        for name,value in model.state_dict().items():
            if not torch.equal(value.cpu(),state[name]):raise ValueError('Frozen/pre-update parameters changed: '+name)

    def group_wake(model,prefixes,zero):
        values={}
        for name,p in model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise ValueError('Missing/nonfinite gradient: '+name)
            values[name]=float(p.grad.abs().max())
        for prefix in prefixes:
            maximum=max(v for n,v in values.items() if n.startswith(prefix))
            if (zero and maximum!=0) or (not zero and maximum<=0):raise ValueError('Unexpected wake state: '+prefix)
        return values

    def compare_Adam(a,b):
        aa=a.state_dict()['state'];bb=b.state_dict()['state']
        if aa.keys()!=bb.keys():raise ValueError('Adam state ownership differs')
        errors=[]
        for key in aa:
            for field in ('step','exp_avg','exp_avg_sq'):
                errors.append(compare(torch,aa[key][field].cpu(),bb[key][field].cpu(),'gradients_moments'))
        return max(errors)

    def path_reference(bank,members,include_residual=True):
        refs={};before=cpu_tree(bank.state_dict());before_rng=rng_snapshot()
        def observe(m,path,opt,logits,peers,train_ids,train_labels,residual):
            unchanged(bank,before)  # No earlier route changes before all gradients.
            shadow=copy.deepcopy(path);shadow_opt=adam(shadow.parameters())
            shadow_opt.load_state_dict(cpu_tree(opt.state_dict()));shadow_opt.zero_grad(set_to_none=True)
            p=torch.softmax(logits.detach(),1)[train_ids]
            target=torch.nn.functional.one_hot(train_labels,10).to(p.dtype)
            q=(peers[:,train_ids].detach().sum(0)+p)/4;w=q-target
            cotangent=torch.zeros_like(logits)
            cotangent[train_ids]=(p-target+(p*(w-(p*w).sum(1,keepdim=True)) if residual else 0))/580
            # Manual VJP uses the exact same active graph, not a second stochastic forward.
            gradients=torch.autograd.grad(logits,tuple(path.parameters()),grad_outputs=cotangent,retain_graph=True)
            for p_ref,g in zip(shadow.parameters(),gradients):p_ref.grad=g.detach().clone()
            refs[m]=(shadow,shadow_opt,[g.detach().cpu().clone() for g in gradients])
        measured=bank_epoch(bank,x,edge,ids,labels,members,include_residual=include_residual,gradient_observer=observe)
        if not same_rng(before_rng,rng_snapshot()):raise ValueError('Private reference advanced caller RNG')
        del before
        errors=[]
        for m in members:
            shadow,shadow_opt,gradients=refs[m]
            for (name,p),g in zip(bank.paths[m].named_parameters(),gradients):
                errors.append(compare(torch,p.grad.cpu(),g,'gradients_moments'))
            shadow_opt.step()
            for name,value in bank.paths[m].state_dict().items():
                compare(torch,value,shadow.state_dict()[name],'parameters')
            compare_Adam(bank.optimizers[m],shadow_opt)
        return {'same_forward_manual_VJP':True,'populated_Adam_reference':True,'gradient_max_abs':max(errors),
            'all_updates_after_gradient_collection':True,'caller_RNG_unchanged':True,'TRAIN':measured}

    graph_groups=['graph.raw.','graph.h_lin.','graph.conv.','graph.root_lin.','graph.ln.','graph.beta']
    single_checks=[]
    try:
        for mode in (False,True):
            native,opt,stream=construct_native(17,x.device);native._global=mode;native.eval()
            native_state=cpu_tree(native.state_dict());outputs=[]
            head=native.pred_global if mode else native.pred_local
            handle=head.register_forward_hook(lambda module,arguments,value:outputs.append(value))
            try:
                with torch.no_grad():h,captured=native_hidden_logits(native,x,edge)
            finally:handle.remove()
            if len(outputs)!=1 or outputs[0] is not captured:raise ValueError('Same-call actual native head identity')
            bank=ResidualBank([native],17,x.device);bank.cache=[(h.detach().clone(),captured.detach().clone())]
            initial=bank.serving(x,edge)
            for m in range(4):
                if not torch.equal(initial[m],torch.softmax(captured,1)):raise ValueError('Literal captured native inclusion')
            record={'stage_global':mode,'same_call_native_inclusion':True,'private_references':[]}
            for m in range(4):
                first=path_reference(bank,[m]);first['groups']=group_wake(bank.paths[m],graph_groups,True)
                if max(float(p.grad.abs().max()) for p in bank.paths[m].output.parameters())<=0:raise ValueError('Output cannot learn first')
                second=path_reference(bank,[m]);second['groups']=group_wake(bank.paths[m],graph_groups,False)
                unchanged(native,native_state)
                if any(p.grad is not None for p in native.parameters()):raise ValueError('Frozen donor gradient')
                record['private_references'].append({'member':m,'first':first,'after_wake':second})
                deadline(started,job)
            # Actual interleaved route gradients now see asymmetric, nonzero learned peers.
            peers=bank.serving(x,edge)
            if torch.equal(peers[0],peers[1]) or torch.equal(peers[0],initial[0]):raise ValueError('No asymmetric nonzero peer fixture')
            record['asymmetric_interleaved']=path_reference(bank,list(range(4)))
            record['own_only_reference']=path_reference(bank,[0],include_residual=False)
            results['checks'].append(record)
            del bank,native,opt,stream,native_state,outputs,head,h,captured,initial,peers,record
            torch.cuda.empty_cache();deadline(started,job)
            # Full capable single: two ordinary/checkpoint comparisons per native mode.
            donor,opt,stream=construct_native(17,x.device);donor._global=mode
            single=PathSingle(donor,opt.state_dict(),stream,17,x.device)
            head_outputs=[];head=donor.pred_global if mode else donor.pred_local
            handle=head.register_forward_hook(lambda module,arguments,value:head_outputs.append(value))
            try:
                with torch.no_grad():single.eval();included=single.serving(x,edge)[0]
            finally:handle.remove()
            if len(head_outputs)!=1 or not torch.equal(included,torch.softmax(head_outputs[0],1)):
                raise ValueError('Capable single same-call native inclusion')
            single_record={'stage_global':mode,'initial_same_call_inclusion':True,'updates':[]}
            for iteration in range(2):
                shadow=copy.deepcopy(single);before_rng=rng_snapshot();actual_forward={};reference_forward={}
                optimizer_ownership(shadow,[shadow.native_optimizer,shadow.new_optimizer])
                def actual_observe(z,loss):actual_forward.update(logits=cpu_tree(z),loss=cpu_tree(loss))
                def reference_observe(z,loss):reference_forward.update(logits=cpu_tree(z),loss=cpu_tree(loss))
                actual=single_epoch(single,x,edge,ids,labels,checkpointed=True,forward_observer=actual_observe)
                if not same_rng(before_rng,rng_snapshot()):raise ValueError('Checkpointed single caller RNG changed')
                reference=single_epoch(shadow,x,edge,ids,labels,checkpointed=False,forward_observer=reference_observe)
                if not same_rng(before_rng,rng_snapshot()):raise ValueError('Ordinary single caller RNG changed')
                logit_error=compare(torch,actual_forward['logits'],reference_forward['logits'],'logits')
                loss_error=compare(torch,actual_forward['loss'],reference_forward['loss'],'logits')
                gradient_errors=[];parameter_errors=[]
                for (name,p),(rname,rp) in zip(single.named_parameters(),shadow.named_parameters()):
                    if name!=rname:raise ValueError('Single parameter identity changed')
                    is_active=not name.startswith('donor.') or active_native(name[6:],mode)
                    if is_active:gradient_errors.append(compare(torch,p.grad.cpu(),rp.grad.cpu(),'gradients_moments'))
                    elif p.grad is not None or rp.grad is not None:raise ValueError('Inactive single native parameter has gradient')
                    parameter_errors.append(compare(torch,p,rp,'parameters'))
                mom_error=max(compare_Adam(single.native_optimizer,shadow.native_optimizer),compare_Adam(single.new_optimizer,shadow.new_optimizer))
                astreams=[single.base_stream]+single.streams;rstreams=[shadow.base_stream]+shadow.streams
                if len(astreams)!=5 or any(not torch.equal(a['cpu'],b['cpu']) or not torch.equal(a['cuda'],b['cuda']) for a,b in zip(astreams,rstreams)):
                    raise ValueError('Five native/private route streams differ between checkpoint/ordinary')
                if max(float(p.grad.abs().max()) for p in single.readout[-1].parameters())<=0:raise ValueError('Single output did not learn')
                readout_hidden=max(float(p.grad.abs().max()) for p in single.readout[0].parameters())
                if (iteration==0 and readout_hidden!=0) or (iteration==1 and readout_hidden<=0):raise ValueError('Single nonlinear decoder wake')
                for path in single.paths:
                    group_wake(path,['raw.','h_lin.','conv.','root_lin.','ln.','beta'],iteration==0)
                if max(float(p.grad.abs().max()) for n,p in single.donor.named_parameters() if active_native(n,mode))<=0:
                    raise ValueError('Selected native single mode is not learning')
                single_record['updates'].append({'iteration':iteration,'checkpoint_vs_ordinary_logits_max_abs':logit_error,
                    'loss_max_abs':loss_error,'active_gradient_max_abs':max(gradient_errors),'parameter_max_abs':max(parameter_errors),
                    'Adam_moment_max_abs':mom_error,'all_five_route_streams_exact':True,'caller_RNG_unchanged':True,
                    'output_then_all_graph_and_nonlinear_decoder_group_wake':True,'actual_TRAIN':actual,'reference_TRAIN':reference,
                    'noncheckpointed_reference_memory_not_omitted':True})
                del shadow,actual_forward,reference_forward
                torch.cuda.empty_cache();deadline(started,job)
            single_checks.append(single_record)
            del single,donor,opt,stream,head_outputs,head,included,single_record
            torch.cuda.empty_cache();deadline(started,job)
        donors=[]
        for m in range(4):
            donor,opt,stream=construct_native(17+1009*m,x.device);donor._global=True;donors.append(donor);del opt,stream
        untied=ResidualBank(donors,17,x.device);untied.make_cache(x,edge)
        path_reference(untied,list(range(4)));untied.assert_storage()
        results['untied_independent_donor_ownership']=True;results['single_checkpoint_references']=single_checks
        del untied,donors,donor;torch.cuda.empty_cache();deadline(started,job)
        results.update(passed=True,inclusive_seconds=time.monotonic()-started,program_sha256=sha(__file__),job_sha256=sha(args.job),
            predictive_verdict=None,fixture_all_required_checks_completed=True)
    except BaseException as error:
        write(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'results':results,
            'partial_outputs_preserved':True,'states_discarded':True,'retry':False});raise
    finally:rng_restore(old_rng)
    if not same_rng(old_rng,rng_snapshot()):raise ValueError('Numerical caller RNG restoration failed')
    results['caller_RNG_restored']=True;write(output/'QUALIFICATION.json',results)


if __name__=='__main__':main()
