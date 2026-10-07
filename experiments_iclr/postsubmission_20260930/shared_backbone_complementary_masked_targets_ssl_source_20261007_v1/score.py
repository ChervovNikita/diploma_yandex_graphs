"""Once-only seed endpoint scoring, admitted only after all fifteen fits."""
import json
import time
from common import PHASE,guard,runtime,load_data,bound,write,sha,deadline

def main():
    args,job,out,plan,context=guard(__file__,'score');torch,versions=runtime(context)
    from method import Predictor,BASE_METHOD
    # guard verified the entire15-cell freeze matrix before this VALID load.
    data=load_data(torch,job,False);x,edge=data['x'],data['edge_index'];vid,labels=data['valid_ids'],data['valid_y'];out.mkdir();started=time.monotonic();results=[]
    def metric(p):
        p=p.double();target=torch.nn.functional.one_hot(labels,10).double();correct=int((p.argmax(1)==labels).sum())
        return {'correct':correct,'nodes':len(labels),'accuracy':correct/len(labels),
            'NLL':float(-p[torch.arange(len(labels)),labels].clamp_min(1e-300).log().mean()),'Brier':float((p-target).square().sum(1).mean())}
    try:
        for row in job['endpoint_freezes']:
            if row['seed']!=job['score_seed']:continue
            freeze=json.loads(bound(row['freeze']).read_text());record=json.loads(bound(freeze['donor_freeze']).read_text())
            donor,opt,stream=BASE_METHOD.construct_native(row['seed'],x.device);del opt,stream
            model=Predictor(donor,row['seed'],row['arm'],x.device)
            state=torch.load(bound(freeze['model']),map_location='cpu',weights_only=True)
            if (state['seed'],state['arm'])!=(row['seed'],row['arm']):raise ValueError('Fixed endpoint identity mismatch')
            model.load_state_dict(state['model'],strict=True);donor._global=state['native_global']
            if donor._global!=record['selected_global']:raise ValueError('Frozen donor mode changed')
            before=time.monotonic();model.make_clean_cache(x,edge);probabilities=model.serving(x,edge)
            if not bool(torch.isfinite(probabilities).all()):raise FloatingPointError('Nonfinite actual endpoint serving probabilities')
            torch.cuda.synchronize();serving_seconds=time.monotonic()-before
            values=probabilities.detach().cpu()[:,vid];pool=values.mean(0)
            target=out/(row['arm']+'_VALID_PROBABILITIES.pt')
            torch.save({'members':values,'pool':pool,'endpoint_freeze':row['freeze'],'seed':row['seed'],'arm':row['arm']},target)
            results.append({'seed':row['seed'],'arm':row['arm'],'pool':metric(pool),'members':[metric(p) for p in values],
                'serving_seconds':serving_seconds,'actual_serving_backbone_forwards':1,'actual_serving_private_path_forwards':4,
                'SSL_decoder_forwards':0,'endpoint_freeze':row['freeze'],'probabilities':{'path':str(target.relative_to(PHASE)),'sha256':sha(target)}})
            del model,donor,state,probabilities;deadline(started,job)
        if len(results)!=5:raise ValueError('All five seed endpoints must be scored once')
        write(out/'ENDPOINTS.json',{'complete':True,'all15_fits_frozen_before_VALID_access':True,'results':results,'source_manifest_sha256':job['source_manifest_sha256'],
            'job_sha256':sha(args.job),'plan':job['plan'],'runtime':versions,'physical_gpu_uuid':job['physical_gpu_uuid'],
            'fixed_final_endpoints_no_reselection':True,'TEST_access':False,'retry':False,'inclusive_seconds':time.monotonic()-started})
    except BaseException as error:
        write(out/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'partial_outputs_preserved':True,'TEST_access':False,'retry':False});raise

if __name__=='__main__':main()
