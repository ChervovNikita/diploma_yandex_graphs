"""All six predeclared native-single warm fits; normal in-process GPU/CPU training."""
import argparse
import gc
import json
import sys
import time
import traceback
sys.dont_write_bytecode = True
import common as c
import native_training as n


def run_case(out, split, seed, recipe_id, recipe, x, edge, block, data, release, device):
    import torch
    case = out/f'split{split}_seed{seed}'/recipe_id
    case.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter(); model = optimizer = packed = best_live = None
    row = {'split':split,'optimizer_seed':seed,'recipe_id':recipe_id,'attempted':True}
    try:
        n.seed_all(seed, device)
        roles = c.fit_control(block, split, seed)
        role_receipt = {'official_train_ids':list(block['train_ids']),'fit_ids':list(roles.fit),
                        'control_ids':list(roles.control),'role_seed':f'condresp-amazon-v1|split={split}|opt={seed}|roles',
                        'rounding':'floor(4*n_class/5) fit, remaining control',
                        'official_train_labels':data['train_labels'][str(split)],
                        'validation_labels':data['validation_labels'][str(split)],
                        'control_is_supervision_not_heldout':True}
        c.write(case/'ROLES.json', role_receipt)
        packed_cpu, token_receipt = n.tokens(x,edge,recipe,data,release['runtime_receipt'])
        packed = packed_cpu.to(device); del packed_cpu
        model, optimizer, names, spec = n.build(recipe, x.shape[1], seed, device)
        bindings = {'split':split,'optimizer_seed':seed,'recipe_id':recipe_id,'recipe':recipe,'spec':spec,
                    'optimizer_names':list(names),'roles':c.record(case/'ROLES.json'),
                    'optimizer_constructor_options':n.adam_constructor_options(optimizer),
                    'data_manifest':release['data_manifest'],'packet_manifest':release['packet_manifest'],
                    'runtime_receipt':release['runtime_receipt'],'tokens':token_receipt,
                    'members':1,'R_S_fixed_at_one':True,'optimizer':'source Adam, coupled L2 weight decay',
                    'serving':'one native predictor, native graph logits', 'TEST_label_use_or_scoring':False}
        c.write(case/'BINDINGS.json', bindings)
        best_accuracy = 0.0; bad_counter = 0; selected = None; selected_logits = None
        c.write(case/'STARTED.json', {'UTC':c.utc(),'bindings':c.record(case/'BINDINGS.json')})
        if device.startswith('cuda:'):
            torch.cuda.reset_peak_memory_stats(device)
        with (case/'TRACE.jsonl').open('x') as trace:
            for epoch in range(1,2001):
                update_start = time.perf_counter()
                loss = n.train_step(model, optimizer, packed, roles)
                logits = n.forward_eval(model, packed)
                metrics = {'fit':n.score(logits,roles.fit,roles.labels_for('fit')),
                           'control':n.score(logits,roles.control,roles.labels_for('control')),
                           'validation':n.score(logits,block['validation_ids'],block['validation_labels'])}
                accuracy = metrics['validation']['accuracy']
                improvement = accuracy > best_accuracy
                if improvement:
                    best_accuracy = accuracy; bad_counter = 0
                    selection = {'epoch':epoch,'validation_accuracy':accuracy,
                                 'selector':'strict validation accuracy, earliest exact ties, initial best0',
                                 'metrics':metrics}
                    checkpoint = n.save_checkpoint(case/f'checkpoint_epoch{epoch:04d}.pt', model,optimizer,bindings,selection,device)
                    best_live = n.snapshot_live(model, optimizer, device)
                    selected = {'selection':selection,'checkpoint':checkpoint}
                    selected_logits = logits.index_select(0,torch.tensor(block['validation_ids'],device=logits.device)).detach().cpu().clone()
                else:
                    bad_counter += 1
                del logits
                n.synchronize(device)
                trace.write(json.dumps({'epoch':epoch,'TRAIN_loss_before_update':float(loss),
                        'metrics':metrics,'strict_improvement':improvement,'bad_counter':bad_counter,
                        'checkpoint':checkpoint if improvement else None,
                        'elapsed_seconds':time.perf_counter()-update_start,'memory':n.memory(device)},allow_nan=False)+'\n')
                trace.flush()
                if bad_counter == 250:
                    break
        row.update(epochs=epoch,stopping_reason='patience250' if bad_counter==250 else 'max_epochs2000',
                   trace=c.record(case/'TRACE.jsonl'),roles=c.record(case/'ROLES.json'),bindings=c.record(case/'BINDINGS.json'),
                   best_validation_accuracy=best_accuracy)
        if selected is None:
            # Preserve the source selector's initial0 semantics. Never fabricate
            # a zero-accuracy selected checkpoint or discard this weak recipe.
            row.update(status='completed_no_strict_positive_selection',completed=True,
                       selected_checkpoint_present=False,comparison_eligible=False)
        else:
            replay = n.replay_selected(best_live,selected['checkpoint'],recipe,x.shape[1],seed,device,packed,roles)
            c.write(case/'REPLAY.json', replay)
            with (case/'selected_validation_logits.pt').open('xb') as f:
                torch.save({'ids':torch.tensor(block['validation_ids'],dtype=torch.int64),
                            'logits':selected_logits,'checkpoint':selected['checkpoint']},f)
            selected.update(validation_logits=c.record(case/'selected_validation_logits.pt'),replay=c.record(case/'REPLAY.json'))
            c.write(case/'SELECTION.json', selected)
            row.update(status='selected',completed=True,comparison_eligible=True,
                       selection=c.record(case/'SELECTION.json'),checkpoint=selected['checkpoint'],
                       validation_logits=selected['validation_logits'],replay=selected['replay'])
        row['memory'] = n.memory(device)
    except Exception as error:
        row.update(status='failed',completed=False,error_type=type(error).__name__,message=str(error),
                   traceback=traceback.format_exc(),comparison_eligible=False,automatic_retry=False)
    finally:
        del model, optimizer, packed, best_live
        gc.collect()
        if device.startswith('cuda:'):
            torch.cuda.empty_cache()
    row.update(elapsed_seconds=time.perf_counter()-begin,TEST_labels_used_or_scored=False)
    c.write(case/'TERMINAL.json', row)
    return dict(row,terminal=c.record(case/'TERMINAL.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',required=True);p.add_argument('--data',required=True)
    p.add_argument('--device',required=True);p.add_argument('--output',required=True)
    args = p.parse_args()
    release, released = c.admission(args.admission,'training',args.output,data=args.data)
    resource = c.read(c.verify(release['resource_receipt']))
    c.require(release['root_observed_resource_pass'] is True and resource['status']=='passed' and
              resource['packet_manifest']==release['packet_manifest'] and resource['data_manifest']==release['data_manifest'] and
              resource['runtime_receipt']==release['runtime_receipt'] and resource['device']==args.device,
              'Exact successful root-observed both-recipe representative resource receipt required')
    out = c.fresh_directory(args.output);started=time.perf_counter();rows=[];errors=[];preserved=False
    c.write(out/'STUDY_STARTED.json',{'UTC':c.utc(),'release':released,'expected_cases':[list(x) for x in c.EXPECTED_CASES],
             'both_recipes_kept_even_if_weak':True,'automatic_retry':False,'normal_process_no_isolation_wrapper':True})
    try:
        torch, actual = c.runtime(args.device,release)
        c.write(out/'RUNTIME.json',actual)
        data,x,edge,blocks=c.load_public(args.data,with_validation=True)
        design=c.read(c.PACKET/'DESIGN.json')
        for split,seed,recipe_id in c.EXPECTED_CASES:
            rows.append(run_case(out,split,seed,recipe_id,design['recipes'][recipe_id],x,edge,blocks[split],data,release,args.device))
        c.preserved(data,release);preserved=True
    except Exception as error:
        errors.append({'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    finally:
        seen={(r['split'],r['optimizer_seed'],r['recipe_id']):r for r in rows}
        rows=[seen.get(key,dict(split=key[0],optimizer_seed=key[1],recipe_id=key[2],status='blocked',
                attempted=False,completed=False,comparison_eligible=False,TEST_labels_used_or_scored=False)) for key in c.EXPECTED_CASES]
        complete=not errors and preserved and all(r['completed'] for r in rows)
        eligible=complete and all(r['comparison_eligible'] for r in rows)
        c.write(out/'STUDY.json',{'schema':'amazon_native_warm_complete6_v1','UTC':c.utc(),'rows':rows,'errors':errors,
                'all6_closed':complete,'all6_selected_replayed':eligible,'originals_preserved':preserved,
                'release':released,'packet_manifest':release['packet_manifest'],'data_manifest':release['data_manifest'],
                'runtime_receipt':release['runtime_receipt'],'device':args.device,'elapsed_seconds':time.perf_counter()-started,
                'cross_recipe_comparison_executed':False,'scientific_competence_gate':'unresolved_no_comparable_published_Amazon_reference',
                'TEST_labels_used_or_scored':False,'automatic_restart_or_retry':False})
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
