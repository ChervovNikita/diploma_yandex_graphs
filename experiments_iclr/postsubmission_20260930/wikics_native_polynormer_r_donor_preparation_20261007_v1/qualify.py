#!/usr/bin/env python3
"""Disabled unchanged-native full-TRAIN both-mode gradient/Adam and epoch costs."""
import random
import time
from common import guard,runtime,load_data,write,sha,deadline,TOLERANCES

def main():
    args,job,output=guard(__file__,'native_qualification');torch,versions=runtime()
    import numpy as np
    from native_engine import construct,epoch
    data,authority=load_data(torch,job,True)
    x,edge,ids,labels=(data[k] for k in ('x','edge_index','train_ids','train_y'))
    output.mkdir();started=time.monotonic()
    old=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    results={'kind':'native_qualification','source_manifest_sha256':job['source_manifest_sha256'],
        'runtime':versions,'TRAIN_only':True,'VALID_values_access':False,'TEST_access':False,'states_discarded':True,
        'full1100_fits':0,'thresholds':TOLERANCES,'no_family_or_repeat_forward_requirement':True,'stages':[]}
    try:
        for global_stage in (False,True):
            before=time.monotonic();model,opt,stream=construct(17,x.device);model._global=global_stage
            torch.cuda.synchronize();setup=time.monotonic()-before
            checked=[]
            for iteration in range(2):
                checked.append(epoch(model,opt,stream,x,edge,ids,labels,reference=True));deadline(started,job)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=time.monotonic()
            measured=epoch(model,opt,stream,x,edge,ids,labels,reference=False)
            torch.cuda.synchronize()
            results['stages'].append({'global_stage':global_stage,'setup_seconds':setup,'same_forward_native_gradient_Adam_checks':checked,
                'complete_epoch_seconds':time.monotonic()-before,'complete_epoch':measured,
                'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),
                'global_cost_limit':'Fresh global-mode native state, no local100/global1000 convergence or predictive claim.' if global_stage else 'Fresh local state.'})
            write(output/'PROGRESS.json',{'completed_stages':len(results['stages']),'target':2,'TRAIN_only':True})
            del model,opt,stream,checked,measured;torch.cuda.empty_cache();deadline(started,job)
        results.update(passed=True,program_sha256=sha(__file__),job_sha256=sha(args.job),inclusive_seconds=time.monotonic()-started,predictive_verdict=None)
    except BaseException as error:
        write(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'results':results,
            'partial_outputs_preserved':True,'retry':False,'VALID_values_access':False,'TEST_access':False});raise
    finally:
        random.setstate(old[0]);np.random.set_state(old[1]);torch.set_rng_state(old[2]);torch.cuda.set_rng_state(old[3])
    current=np.random.get_state()
    if random.getstate()!=old[0] or current[0]!=old[1][0] or not np.array_equal(current[1],old[1][1]) or current[2:]!=old[1][2:] or not torch.equal(torch.get_rng_state(),old[2]) or not torch.equal(torch.cuda.get_rng_state(),old[3]):
        raise ValueError('Qualification caller RNG restoration')
    results['caller_RNG_restored']=True;write(output/'QUALIFICATION.json',results)

if __name__=='__main__':main()
