"""Fifteen fixed TRAIN-only raw-feature SSL/classification fits; no held scoring."""
import gzip
import json
import resource
import time
from common import PHASE,guard,runtime,load_data,bound,write,sha,deadline

def main():
    args,job,out,plan,context=guard(__file__,'fit');torch,versions=runtime(context)
    from method import Predictor,SSL_ARMS,start_ssl,end_ssl,ssl_update,mask_block,adjacency,bank_class_step,single_class_step,BASE_METHOD
    data=load_data(torch,job,True);x,edge,ids,labels=(data[k] for k in ('x','edge_index','train_ids','train_y'));out.mkdir();started=time.monotonic()
    def row(path):return {'path':str(path.relative_to(PHASE)),'sha256':sha(path)}
    counters={'masked_backbone_forwards':0,'SSL_AdamW_updates':0,'SSL_private_forward_backward':0,'SSL_decoder_forward_backward':0,
        'clean_class_backbone_forwards':0,'class_path_forward_backward':0,'class_optimizer_updates':0,'frozen_peer_path_forwards':0,'serving_backbone_forwards':0,'serving_private_path_forwards':0}
    try:
        torch.cuda.reset_peak_memory_stats();record=json.loads(bound(job['donor_freeze']).read_text());state=torch.load(bound(record['selected']),map_location='cpu',weights_only=True)
        donor,opt,stream=BASE_METHOD.construct_native(job['seed'],x.device);donor.load_state_dict(state['model']);donor._global=state['stage_global']
        if state['epoch']!=record['selected_epoch'] or donor._global!=record['selected_global']:raise ValueError('Selected native epoch/mode mismatch')
        del state,opt,stream
        model=Predictor(donor,job['seed'],job['arm'],x.device)
        torch.save({'model':BASE_METHOD.cpu_tree(model.state_dict()),'streams':BASE_METHOD.cpu_tree(model.streams),'arm':job['arm'],'seed':job['seed']},out/'INITIAL_STATE.pt')
        ssl_started=time.monotonic()
        if job['arm'] in SSL_ARMS:
            ssl_opt,parameters=start_ssl(model,job['seed'],x.device);adj=adjacency(edge,len(x));generator=torch.Generator(device='cpu');generator.manual_seed(job['seed']+940001)
            with (out/'SSL_HISTORY.jsonl').open('x') as history,gzip.open(out/'MASK_DRAWS.jsonl.gz','wt',encoding='utf-8') as draws:
                for block in range(25):
                    before=time.monotonic();xmask,masked,quarters,eligible=mask_block(x,ids,generator)
                    hmask,_unused_masked_logits=model.encode(xmask,edge);del _unused_masked_logits
                    counters['masked_backbone_forwards']+=1
                    draws.write(json.dumps({'block':block,'eligible_U':eligible,'quarters':quarters.cpu().tolist(),
                        'repeated_assignment':'Qt for every m','complementary_assignment':'Q((m+t)%4)'})+'\n');draws.flush()
                    block_targets=[0]*4
                    for t in range(4):
                        result=ssl_update(model,ssl_opt,parameters,xmask,hmask,edge,masked,quarters,t,x,adj)
                        counters['SSL_AdamW_updates']+=1;counters['SSL_private_forward_backward']+=4;counters['SSL_decoder_forward_backward']+=4
                        for m,n in enumerate(result['branch_target_counts']):block_targets[m]+=n
                        history.write(json.dumps({'block':block,'update':t,'TRAIN_unlabeled_SSL':result})+'\n');history.flush();deadline(started,job)
                    if block_targets!=[len(masked)]*4:raise ValueError('Each branch must see each masked target once perblock')
                    model.assert_zero_classes();del xmask,hmask,masked,quarters
                    write(out/'PROGRESS.json',{'arm':job['arm'],'SSL_blocks':block+1,'SSL_target_blocks':25,'counters':counters,'block_seconds':time.monotonic()-before,'held_values_accessed':False})
            torch.save({'decoders':BASE_METHOD.cpu_tree(model.decoders.state_dict()),'SSL_optimizer':BASE_METHOD.cpu_tree(ssl_opt.state_dict()),
                'class_heads_zero':True,'decoder_parameters':615600},out/'SSL_DECODERS_DISCARDED.pt')
            del ssl_opt,parameters,adj;end_ssl(model)
        ssl_seconds=time.monotonic()-ssl_started
        model.assert_zero_classes();clean_started=time.monotonic();model.make_clean_cache(x,edge);torch.cuda.synchronize()
        clean_encoding_seconds=time.monotonic()-clean_started;counters['clean_class_backbone_forwards']+=1
        h0,z0=model.clean_cache
        frozen_train_CE=float(torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(z0,1)[ids],labels))
        write(out/'SSL_FREEZE.json',{'complete':True,'arm':job['arm'],'SSL_used':job['arm'] in SSL_ARMS,'counters':counters,
            'SSL_seconds':ssl_seconds,'class_outputs_zero_through_SSL':True,'raw_targets_only':True,'teacher_forwards':0,
            'clean_H0_z0_allowed_only_after_SSL':True,'decoder_parameters_active_when_SSL':615600 if job['arm'] in SSL_ARMS else 0,
            'classification_Adam_reset':True,'frozen_native_TRAIN_CE':frozen_train_CE,'VALID_values_access':False,'TEST_access':False})
        class_started=time.monotonic()
        with (out/'CLASS_HISTORY.jsonl').open('x') as history:
            if model.single:
                parameters=list(model.paths.parameters())+list(model.classifier.parameters());optimizer=BASE_METHOD.adam(parameters)
                for t in range(100):
                    before=time.monotonic();result=single_class_step(model,optimizer,x,edge,ids,labels)
                    counters['class_path_forward_backward']+=4;counters['class_optimizer_updates']+=1
                    history.write(json.dumps({'joint_update':t+1,'TRAIN':result,'seconds':time.monotonic()-before})+'\n');history.flush();deadline(started,job)
                    write(out/'PROGRESS.json',{'arm':job['arm'],'class_joint_updates':t+1,'target':100,'counters':counters,'held_values_accessed':False})
            else:
                optimizers=[BASE_METHOD.adam(list(model.paths[m].parameters())+list(model.classifiers[m].parameters())) for m in range(4)]
                for m in range(4):
                    model.eval()
                    with torch.no_grad():peers=torch.stack([torch.softmax(model.logits(j,x,edge),1) for j in range(4) if j!=m])
                    counters['frozen_peer_path_forwards']+=3
                    for t in range(100):
                        before=time.monotonic();result=bank_class_step(model,m,optimizers[m],x,edge,ids,labels,peers)
                        counters['class_path_forward_backward']+=1;counters['class_optimizer_updates']+=1
                        history.write(json.dumps({'stage':m,'update':t+1,'TRAIN':result,'seconds':time.monotonic()-before})+'\n');history.flush();deadline(started,job)
                        write(out/'PROGRESS.json',{'arm':job['arm'],'class_stage':m,'stage_updates':t+1,'target_perpath':100,'counters':counters,'held_values_accessed':False})
                    del peers
        class_seconds=time.monotonic()-class_started
        if counters['class_path_forward_backward']!=400:raise ValueError('Full four-path class budget required')
        if job['arm'] in SSL_ARMS and (counters['masked_backbone_forwards'],counters['SSL_AdamW_updates'],counters['SSL_private_forward_backward'],counters['SSL_decoder_forward_backward'])!=(25,100,400,400):raise ValueError('Complete fixed SSL budget required')
        endpoint=out/'FINAL_STATE.pt';torch.save({'model':BASE_METHOD.cpu_tree(model.state_dict()),'streams':BASE_METHOD.cpu_tree(model.streams),
            'native_global':donor._global,'arm':job['arm'],'seed':job['seed'],'clean_cache_not_in_state':True},endpoint)
        write(out/'FREEZE.json',{'complete':True,'arm':job['arm'],'seed':job['seed'],'model':row(endpoint),'training_job':row(args.job),
            'donor_freeze':job['donor_freeze'],'donor_origin':job['donor_origin'],'execution_context':job['execution_context'],
            'data_manifest':job['data_manifest'],'source_manifest_sha256':job['source_manifest_sha256'],'program_sha256':sha(__file__),'plan':job['plan'],
            'class_path_updates':400,'counters':counters,'SSL_seconds':ssl_seconds,'class_seconds':class_seconds,
            'clean_class_encoding_seconds':clean_encoding_seconds,
            'inclusive_seconds':time.monotonic()-started,'native_acquisition_seconds_to_charge':record['inclusive_seconds'],
            'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'output_bytes_before_freeze':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            'runtime':versions,'physical_gpu_uuid':job['physical_gpu_uuid'],'parameter_count_served_model':sum(p.numel() for p in model.parameters()),
            'serving_work_per_prediction':{'clean_backbone_forwards':1,'private_graph_paths':4,'SSL_decoders':0},
            'VALID_values_access':False,'TEST_access':False,'endpoint_scored':False,'retry':False,'development_novelty_unverified':True})
    except BaseException as error:
        write(out/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'counters':counters,'partial_outputs_preserved':True,'retry':False,'VALID_values_access':False,'TEST_access':False});raise

if __name__=='__main__':main()
