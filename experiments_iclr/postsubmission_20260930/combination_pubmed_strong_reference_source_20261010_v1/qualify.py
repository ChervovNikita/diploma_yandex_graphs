"""Disabled new-interface TRAIN-only qualification, one fresh update per body."""
import argparse
import json
from pathlib import Path
import resource
import time
from admission import admit_engineering
from source import write
from train import runtime_and_data,timed,add,check_bounds


def qualify(spec,output,started):
    output.mkdir(parents=True,exist_ok=False);timings={};rows=[];session=None
    try:
        np,torch,providers,tensor,roles=runtime_and_data(spec,valid=False)
        from adapter import fresh_single,assembled_I4
        states=[]
        for body_spec in spec['body_roster']:
            check_bounds(spec,output,started)
            session,clock=timed(torch,lambda:fresh_single(spec['condition'],body_spec['initialization_seed'],tensor));add(timings,'construction_preprocessing',clock)
            training,clock=timed(torch,lambda:session.train_step(audit=False));add(timings,'TRAIN_update',clock)
            (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'TRAIN_factual_serving',clock)
            views=4 if spec['condition']=='single_mean4_dropout' else 1
            if session.counters != dict(updates=1,factual_forwards=views+1,masked_forwards=0,backwards=views,optimizer_steps=1,serving_forwards=1,preprocessing_banks=1):
                raise ValueError('Exact engineering counters required')
            rows.append(dict(**body_spec,updates=1,TRAIN_label_count=11829,complete_graph_nodes=19717,
                             counters=dict(session.counters),training=training,
                             preprocessing_seconds=session.preparation_seconds))
            if spec['condition']=='independent4_own':
                states.append({k:v.detach().cpu().clone() for k,v in session.bodies[0].state_dict().items()})
                del session,probabilities,logits;session=None
        assembly=None
        if spec['condition']=='independent4_own':
            session,clock=timed(torch,lambda:assembled_I4(spec['seed'],tensor,states));add(timings,'assembly_construction_preprocessing_restore',clock)
            (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'assembly_serving',clock)
            assembly=dict(counters=dict(session.counters),selected_or_quality_claimed=False)
            if session.counters != dict(updates=0,factual_forwards=4,masked_forwards=0,backwards=0,optimizer_steps=0,serving_forwards=4,preprocessing_banks=1):
                raise ValueError('Engineering assembly must never train')
        check_bounds(spec,output,started)
        result=dict(schema='PubMed-strong-reference-TRAIN-only-engineering-v1',complete=True,
                    condition=spec['condition'],seed=spec['seed'],bodies=rows,assembly=assembly,
                    source_manifest_sha256=spec['source_manifest_sha256'],engineering_release_sha256=spec['_release_sha256'],
                    data_SHA=spec['train_bundle']['sha256'],providers=providers,timings=timings,
                    inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                    CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                    VALID_access=False,TEST_access=False,science_enabled=False,accuracy_evidence=False,
                    paper_score_recalculation=False,automatic_retry=False,owner_success_not_inferred=True)
        write(output/'COMPLETE.json',result);return result
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),bodies=rows,timings=timings,
            inclusive_seconds=time.monotonic()-started,VALID_access=False,TEST_access=False,science_enabled=False,automatic_retry=False))
        raise


def main():
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();spec,output=admit_engineering(args.release,args.release_sha256)
    result=qualify(spec,output,started)
    print(json.dumps(dict(complete=result['complete'],condition=spec['condition'],owner_success_not_inferred=True)))


if __name__=='__main__':main()
