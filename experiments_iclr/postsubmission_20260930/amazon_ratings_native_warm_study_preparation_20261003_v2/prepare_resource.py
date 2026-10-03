"""Actual complete-Amazon resource/native-replay preparation; never auto-launches training."""
import argparse
import gc
import sys
import time
import traceback
sys.dont_write_bytecode = True
import common as c
import native_training as n
import byte_identity as b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission', required=True); p.add_argument('--data', required=True)
    p.add_argument('--device', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    release, released = c.admission(args.admission, 'resource', args.output, data=args.data)
    out = c.fresh_directory(args.output); started = time.perf_counter(); rows = []
    c.write(out/'STARTED.json', {'UTC': c.utc(), 'release': released, 'representative_block': [0,17],
                              'recipes': list(c.RECIPES), 'full_training_launched': False})
    data = actual = byte_checks = None
    try:
        torch, actual = c.runtime(args.device, release)
        import numpy as np
        byte_checks = b.qualify(torch_module=torch, numpy_module=np)
        data, x, edge, blocks = c.load_public(args.data, with_validation=False)
        roles = c.fit_control(blocks[0], 0, 17)
        design = c.read(c.PACKET/'DESIGN.json')
        for recipe_id in c.RECIPES:
            case = out/recipe_id; case.mkdir(); recipe = design['recipes'][recipe_id]
            model = optimizer = packed = live = None; begin = time.perf_counter()
            try:
                if args.device.startswith('cuda:'):
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(args.device)
                n.seed_all(17, args.device)
                packed_cpu, token_receipt = n.tokens(x, edge, recipe, data, release['runtime_receipt'], qualify_native=True)
                packed = packed_cpu.to(args.device); del packed_cpu
                model, optimizer, names, spec = n.build(recipe, x.shape[1], 17, args.device)
                points = [{'stage':'construction', 'memory':n.memory(args.device)}]
                for step in range(1,7):
                    n.train_step(model, optimizer, packed, roles)
                    n.synchronize(args.device)
                    points.append({'stage':'TRAIN_'+str(step), 'memory':n.memory(args.device)})
                # Complete eval allocation, but no VAL labels are decoded and
                # no predictive quality reduction is performed. Preservation
                # later physically opens the compact label files for hashing.
                prediction = n.forward_eval(model, packed)
                c.require(list(prediction.shape) == [x.shape[0],5], 'Complete eval allocation differs')
                del prediction
                live = n.snapshot_live(model, optimizer, args.device)
                checkpoint = n.save_checkpoint(case/'state.pt', model, optimizer,
                    {'recipe_id':recipe_id,'spec':spec,'optimizer_names':list(names),'data_manifest':release['data_manifest'],
                     'optimizer_constructor_options':n.adam_constructor_options(optimizer),
                     'packet_manifest':release['packet_manifest'],'runtime_receipt':release['runtime_receipt']},
                    {'resource_only_after_TRAIN_updates':6}, args.device)
                replay = n.replay_selected(live, checkpoint, recipe, x.shape[1], 17, args.device, packed, roles)
                n.synchronize(args.device)
                row = {'recipe_id':recipe_id, 'status':'passed','split':0,'optimizer_seed':17,
                       'native_token_qualification':token_receipt,'checkpoint':checkpoint,'replay':replay,
                       'fit_targets':len(roles.fit),'control_targets':len(roles.control),
                       'trainable_native_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'allocated_parameters_including_frozen_adapter_factors':sum(p.numel() for p in model.parameters()),
                       'memory_points':points, 'memory_after_checkpoint_and_replay':n.memory(args.device),
                       'six_training_updates':True,'validation_labels_decoded_or_scored':False,'predictive_quality_scored':False}
            except Exception as error:
                row = {'recipe_id':recipe_id,'status':'failed','error_type':type(error).__name__,
                       'message':str(error),'traceback':traceback.format_exc(),'automatic_retry':False}
            finally:
                del model, optimizer, packed, live
                gc.collect()
                if args.device.startswith('cuda:'):
                    torch.cuda.empty_cache()
            row['elapsed_seconds'] = time.perf_counter()-begin
            rows.append(row); c.write(case/'TERMINAL.json', row)
        c.preserved(data, release)
    except Exception as error:
        rows.append({'status':'setup_or_preservation_failed','error_type':type(error).__name__,
                     'message':str(error),'traceback':traceback.format_exc()})
    passed = len(rows)==2 and [r.get('recipe_id') for r in rows]==list(c.RECIPES) and all(r['status']=='passed' for r in rows)
    c.write(out/'RESOURCE.json', {'schema':'amazon_native_warm_resource_v1','UTC':c.utc(),
            'status':'passed' if passed else 'failed','rows':rows,'release':released,
            'packet_manifest':release['packet_manifest'],'data_manifest':release['data_manifest'],
            'runtime_receipt':release['runtime_receipt'],'device':args.device,'runtime':actual,
            'data_free_byte_identity_qualification':byte_checks,
            'elapsed_seconds':time.perf_counter()-started,'representative_scope':'full graph; block0; both recipes; six updates each',
            'other_blocks_full_training_or_physical_OS_confinement_qualified':False,
            'validation_or_TEST_quality_scored':False,'automatic_retry':False,'training_auto_launched':False})
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
