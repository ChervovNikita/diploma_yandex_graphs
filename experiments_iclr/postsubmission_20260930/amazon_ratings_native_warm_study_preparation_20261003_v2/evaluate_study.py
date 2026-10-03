"""Complete-six closure, native saved-state inference and bounded development table."""
import argparse
import gc
import json
import sys
import time
import traceback
sys.dont_write_bytecode = True
import common as c
import native_training as n
RUN_OUTPUT = None


def inspect_closure(study_path, release):
    study=c.read(study_path);root=c.confined(study_path).parent
    recipes=c.read(c.PACKET/'DESIGN.json')['recipes']
    c.require(release['study']==c.record(study_path) and release['root_observed_training_process_exit'] is True and
              release['root_observed_training_exit_code']==0 and study['schema']=='amazon_native_warm_complete6_v1' and
              study['all6_closed'] is True and study['all6_selected_replayed'] is True and
              study['originals_preserved'] is True and study['TEST_labels_used_or_scored'] is False and
              study['packet_manifest']==release['packet_manifest'] and study['data_manifest']==release['data_manifest'] and
              study['runtime_receipt']==release['runtime_receipt'], 'All-six selected closure and root process-exit observation required')
    c.verify(release['training_process_exit_receipt'])
    c.require([(r['split'],r['optimizer_seed'],r['recipe_id']) for r in study['rows']]==list(c.EXPECTED_CASES),
              'Exact ordered six-case family required')
    cases=[]
    for row in study['rows']:
        case=root/f"split{row['split']}_seed{row['optimizer_seed']}"/row['recipe_id']
        c.require(row['terminal']==c.record(case/'TERMINAL.json') and row['status']=='selected' and
                  row['completed'] is True and row['TEST_labels_used_or_scored'] is False, 'Case terminal mismatch')
        terminal=c.read(c.verify(row['terminal']))
        c.require(terminal=={k:v for k,v in row.items() if k!='terminal'}, 'Study terminal projection differs')
        for field,name in [('selection','SELECTION.json'),('trace','TRACE.jsonl'),('bindings','BINDINGS.json'),
                           ('roles','ROLES.json'),('validation_logits','selected_validation_logits.pt'),('replay','REPLAY.json')]:
            c.require(row[field]==c.record(case/name), 'Canonical case artifact differs')
        selected=c.read(case/'SELECTION.json');replay=c.read(case/'REPLAY.json')
        c.require(selected['checkpoint']==row['checkpoint'] and selected['replay']==row['replay'] and
                  selected['validation_logits']==row['validation_logits'] and replay['selected_logits_bitwise_replayed'] is True and
                  replay['default_Adam_state_dict_RNG_next_step_bitwise_replayed'] is True, 'Selected native/default-Adam state_dict/RNG replay receipt required')
        c.verify(row['checkpoint'])
        bindings=c.read(case/'BINDINGS.json')
        c.require(bindings['split']==row['split'] and bindings['optimizer_seed']==row['optimizer_seed'] and
                  bindings['recipe_id']==row['recipe_id'] and bindings['data_manifest']==release['data_manifest'] and
                  bindings['recipe']==recipes[row['recipe_id']] and
                  bindings['packet_manifest']==release['packet_manifest'] and bindings['runtime_receipt']==release['runtime_receipt'] and
                  bindings['members']==1 and bindings['R_S_fixed_at_one'] is True and bindings['TEST_label_use_or_scoring'] is False,
                  'Checkpoint/native-single authority differs')
        # Independently replay the strict source selector and stopping rule from
        # the complete trace. A partial/favorable recipe subset is never tabled.
        trace=[json.loads(line) for line in (case/'TRACE.jsonl').read_text().splitlines()]
        c.require(trace and len(trace)==row['epochs'] and len(trace)<=2000, 'Incomplete epoch trace')
        best=0.0;bad=0;winner=None
        for epoch,event in enumerate(trace,1):
            acc=event['metrics']['validation']['accuracy'];improve=acc>best
            if improve:
                best=acc;bad=0;winner=event
                c.verify(event['checkpoint'])
            else:bad+=1
            c.require(event['epoch']==epoch and event['strict_improvement']==improve and event['bad_counter']==bad and
                      (improve or event['checkpoint'] is None) and
                      (epoch==len(trace) or bad<250), 'Source accuracy selector/early stopping trace differs')
        c.require(winner is not None and winner['epoch']==selected['selection']['epoch'] and
                  winner['checkpoint']==selected['checkpoint'] and
                  winner['metrics']==selected['selection']['metrics'] and
                  selected['selection']['validation_accuracy']==best and
                  row['best_validation_accuracy']==best and
                  selected['checkpoint']==c.record(case/f"checkpoint_epoch{winner['epoch']:04d}.pt") and
                  ((bad==250 and row['stopping_reason']=='patience250') or
                   (len(trace)==2000 and bad<250 and row['stopping_reason']=='max_epochs2000')),
                  'Earliest strict-accuracy selected checkpoint or source stop differs')
        cases.append((row,bindings,selected))
    return study,cases


def main():
    global RUN_OUTPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',required=True);p.add_argument('--data',required=True);p.add_argument('--study',required=True)
    p.add_argument('--device',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    release,released=c.admission(args.admission,'evaluation',args.output,data=args.data)
    study,cases=inspect_closure(args.study,release)  # Complete admission precedes Torch/data/scoring.
    c.require(study['device']==args.device,'Original training device/runtime required for bitwise selected inference')
    output=c.fresh_directory(args.output);started=time.perf_counter()
    RUN_OUTPUT = output
    c.write(output/'STARTED.json',{'UTC':c.utc(),'release':released,'study':release['study']})
    torch,actual=c.runtime(args.device,release)
    data,x,edge,blocks=c.load_public(args.data,with_validation=True)
    rows=[]
    for row,bindings,selected in cases:
        j,seed,recipe_id=row['split'],row['optimizer_seed'],row['recipe_id'];recipe=bindings['recipe']
        roles=c.fit_control(blocks[j],j,seed)
        roles_saved=c.read(c.verify(bindings['roles']))
        c.require(roles_saved['fit_ids']==list(roles.fit) and roles_saved['control_ids']==list(roles.control), 'Recomputed TRAIN roles differ')
        packed_cpu,token_receipt=n.tokens(x,edge,recipe,data,release['runtime_receipt'])
        c.require(token_receipt==bindings['tokens'],'Native token identity differs')
        packed=packed_cpu.to(args.device);del packed_cpu
        model,optimizer,names,spec=n.build(recipe,x.shape[1],seed,args.device)
        c.require(list(names)==bindings['optimizer_names'] and spec==bindings['spec'] and
                  n.adam_constructor_options(optimizer)==bindings['optimizer_constructor_options'],
                  'Native spec/default-Adam constructor options/group ordering differ')
        checkpoint=torch.load(c.verify(selected['checkpoint']),map_location='cpu',weights_only=True)
        c.require(checkpoint['bindings']==bindings and checkpoint['selection']==selected['selection'],'Saved state authority differs')
        n.restore(model,optimizer,checkpoint,args.device)
        logits=n.forward_eval(model,packed)
        saved=torch.load(c.verify(row['validation_logits']),map_location='cpu',weights_only=True)
        ids=torch.tensor(blocks[j]['validation_ids'],dtype=torch.int64)
        c.require(saved['checkpoint']==selected['checkpoint'] and n.tensor_equal(saved['ids'],ids) and
                  n.tensor_equal(saved['logits'],logits.index_select(0,ids.to(args.device)).cpu()), 'Independent selected VAL-logit replay differs')
        metrics={'fit':n.score(logits,roles.fit,roles.labels_for('fit')),
                 'control':n.score(logits,roles.control,roles.labels_for('control')),
                 'validation':n.score(logits,blocks[j]['validation_ids'],blocks[j]['validation_labels'])}
        c.require(metrics==selected['selection']['metrics'],'Saved selected metrics differ from independent replay')
        rows.append({'split':j,'optimizer_seed':seed,'recipe_id':recipe_id,'selected_epoch':selected['selection']['epoch'],
                     'metrics':metrics,'checkpoint':row['checkpoint'],'native_selected_inference_bitwise_replayed':True,
                     'TEST_labels_used_or_scored':False})
        del model,optimizer,packed,logits,saved,checkpoint
        gc.collect()
        if args.device.startswith('cuda:'):torch.cuda.empty_cache()
    means={r:sum(row['metrics']['validation']['accuracy'] for row in rows if row['recipe_id']==r)/3 for r in c.RECIPES}
    recommendation='roman_mono' if means['roman_mono']>means['source_defaults'] else 'source_defaults'
    c.preserved(data,release);c.verify(release['study']);c.verify_sources()
    c.write(output/'EVALUATION.json',{'schema':'amazon_native_warm_complete6_development_evaluation_v1','UTC':c.utc(),
             'rows':rows,'mean_selected_validation_accuracy':means,'bounded_development_recipe_recommendation':recommendation,
             'choice_rule':'strictly higher three-block mean accuracy; source_defaults wins exact tie',
             'scientific_competence_gate':'unresolved_no_independently_sourced_comparable_published_native_Amazon_reference',
             'competence_certified':False,'M4_warm_or_intervention_authorized':False,
             'release':released,'study':release['study'],'runtime':actual,'elapsed_seconds':time.perf_counter()-started,
             'both_recipes_all_three_blocks_retained':True,'originals_preserved':True,'TEST_labels_used_or_scored':False,
             'limits':['one graph with overlapping native split/seed blocks','development recipe selection reuses validation labels',
                       'TRAIN80/20 is an adaptation to published source recipes','accuracy-selected NLL/F1 are secondary at the same checkpoint',
                       'no independent published comparable Amazon competence threshold established','PyG2.3 author environment parity and complete dependency closure unqualified']})
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if RUN_OUTPUT is not None:
            c.write(RUN_OUTPUT/'FAILED.json',{'UTC':c.utc(),'error_type':type(error).__name__,
                    'message':str(error),'traceback':traceback.format_exc(),'comparison_emitted':False,
                    'automatic_retry':False})
        raise
