#!/usr/bin/env python3
"""Disabled discarded complete TRAIN epoch costs; no VALID or prediction quality."""
import time
from common import guard,runtime,load_data,write,sha,deadline


def main():
    args,job,output=guard(__file__,'cost');torch,versions=runtime()
    from method import (construct_native,native_step,ResidualBank,PathSingle,bank_epoch,single_epoch,rng_snapshot,rng_restore)
    data,authority=load_data(torch,job,True)
    x,edge,ids,labels=(data[k] for k in ('x','edge_index','train_ids','train_y'))
    output.mkdir();started=time.monotonic();old_rng=rng_snapshot()
    results={'kind':'cost','source_manifest_sha256':job['source_manifest_sha256'],'runtime':versions,
        'TRAIN_only':True,'VALID_values_access':False,'TEST_access':False,'states_discarded':True,
        'full1100_fits':0,'public_nodes':11701,'edges':442907,'TRAIN_labels':580,'costs':[]}
    try:
        for mode in (False,True):
            for arm in ('native1','native4','E_stage','E_joint','E_own','S_paths','U_stage'):
                setup=time.monotonic();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                count=4 if arm in ('native4','U_stage') else 1;donors=[];native_optimizers=[];streams=[]
                for m in range(count):
                    donor,opt,stream=construct_native(17+1009*m,x.device);donor._global=mode
                    donors.append(donor);native_optimizers.append(opt);streams.append(stream)
                if arm in ('E_stage','E_joint','E_own','U_stage'):
                    model=ResidualBank(donors,17,x.device);model.make_cache(x,edge)
                    # Fresh nonzero output state is reached only through a discarded TRAIN update.
                    bank_epoch(model,x,edge,ids,labels,list(range(4)),include_residual=arm!='E_own')
                elif arm=='S_paths':
                    model=PathSingle(donors[0],native_optimizers[0].state_dict(),streams[0],17,x.device)
                    single_epoch(model,x,edge,ids,labels)
                else:model=None
                if model is not None:
                    # Frozen native optimizers are not retained as a false memory overhead.
                    del native_optimizers,streams,donors,opt,stream,donor
                torch.cuda.synchronize();setup_seconds=time.monotonic()-setup
                torch.cuda.reset_peak_memory_stats();before=time.monotonic()
                if arm in ('native1','native4'):
                    measured=[native_step(d,o,s,x,edge,ids,labels) for d,o,s in zip(donors,native_optimizers,streams)]
                elif arm=='S_paths':measured=single_epoch(model,x,edge,ids,labels)
                else:measured=bank_epoch(model,x,edge,ids,labels,list(range(4)) if arm=='E_joint' else [0],include_residual=arm!='E_own',diagnostics=True)
                torch.cuda.synchronize();seconds=time.monotonic()-before
                results['costs'].append({'arm':arm,'stage_global':mode,'setup_cache_wake_seconds':setup_seconds,
                    'complete_epoch_seconds':seconds,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'CUDA_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'TRAIN_work':measured,
                    'note':'Fresh discarded states, not1100-warmed convergence; nonzero correction output reached with one complete TRAIN warm step. Measured bank epoch enables the actual fit diagnostic VJP pair(s), conservatively bounding ordinary epochs without them. Native4 retains four models as a storage upper bound relative to serial independent fitting; actual selectors/metrics/CPU transfer/checkpoint/cache I/O remain additional.'})
                write(output/'PROGRESS.json',{'complete_costs':len(results['costs']),'target':14,'TRAIN_only':True})
                del measured,model
                if arm in ('native1','native4'):del donors,native_optimizers,streams,donor,opt,stream
                torch.cuda.empty_cache();deadline(started,job)
        if len(results['costs'])!=14:raise ValueError('All14 complete native/correction epoch costs required')
        results.update(passed=True,inclusive_seconds=time.monotonic()-started,program_sha256=sha(__file__),job_sha256=sha(args.job),
            predictive_verdict=None,resource_forecast='Root must charge full1100 donors,400 correction routes, peer serving, snapshots, selection and rematerialization; this does not measure complete convergence.')
    except BaseException as error:
        write(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'results':results,
            'partial_outputs_preserved':True,'states_discarded':True,'retry':False});raise
    finally:rng_restore(old_rng)
    import numpy as np
    restored=rng_snapshot()
    if restored[0]!=old_rng[0] or restored[1][0]!=old_rng[1][0] or not np.array_equal(restored[1][1],old_rng[1][1]) or restored[1][2:]!=old_rng[1][2:] or not torch.equal(restored[2],old_rng[2]) or not torch.equal(restored[3],old_rng[3]):
        raise ValueError('Cost qualifier caller RNG restoration failed')
    results['caller_RNG_restored']=True;write(output/'QUALIFICATION.json',results)


if __name__=='__main__':main()
