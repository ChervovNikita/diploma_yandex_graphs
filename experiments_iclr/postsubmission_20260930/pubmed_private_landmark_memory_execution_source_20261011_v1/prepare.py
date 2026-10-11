"""Admitted custodian projection and label-free canonical landmark/BFS capsule."""
from collections import deque
import json
from pathlib import Path
import time
from runtime import HERE,NOMINATION_SHA,SEEDS,Runtime,checked,receipt,require,source_surface,write

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None
    result=dict(stage='prepare',complete=False,retry=False,config=config['_config_receipt'],engineering_policy=config['engineering_policy'],nomination_manifest_sha256=NOMINATION_SHA,source_manifest=config['source_manifest'],custody=custody,TEST_read=False,model_constructed=False,scientific_fits=0,predictions_read=False,CAL_DEVELOPMENT_projection_only=True)
    try:
        rt=Runtime(config,prepared=False);np,torch=rt.np,rt.t;data=rt.binding['server_data_and_runtime']
        valid_custody=json.loads(checked(data['validation_custody']).read_text())
        with np.load(checked(data['valid_bundle']),allow_pickle=False) as archive:
            require(set(archive.files)=={'valid_ids','valid_y'},'Only original VALID projection source');valid_ids,valid_y=archive['valid_ids'].copy(),archive['valid_y'].copy()
        with source_surface() as (engine,factory):
            fingerprint=engine.fingerprint
            require({k:fingerprint(v) for k,v in dict(valid_ids=valid_ids,valid_y=valid_y).items()}==valid_custody['array_fingerprints'],'Exact original VALID arrays')
        require(valid_ids.shape==valid_y.shape==(3942,) and valid_ids.dtype==valid_y.dtype==np.int64 and len(np.unique(valid_ids))==3942 and np.bincount(valid_y,minlength=3).tolist()==[820,1547,1575],'Complete original stratified VALID')
        require(not np.intersect1d(valid_ids,rt.tensor['train_ids'].numpy()).size,'Original TRAIN/VALID disjoint')
        outside=rt.fp(rt.rng());generator=np.random.RandomState(491003011);before=generator.get_state();assignment={name:[] for name in ('select','cal','dev')};permutations=[]
        for label,(selected,calibrated) in enumerate(zip([273,515,525],[273,515,525])):
            positions=np.flatnonzero(valid_y==label);permutation=generator.permutation(len(positions));permutations.append(torch.from_numpy(permutation.copy()))
            # Original VALID order within each class; concatenate class slices.
            ordered=positions[permutation]
            assignment['select'].append(ordered[:selected]);assignment['cal'].append(ordered[selected:selected+calibrated]);assignment['dev'].append(ordered[selected+calibrated:])
        after=generator.get_state();require(rt.fp(rt.rng())==outside,'Isolated role projection changed global RNG')
        role_files={};role_hashes={};all_positions=[]
        for role,counts in (('select',[273,515,525]),('cal',[273,515,525]),('dev',[274,517,525])):
            positions=np.concatenate(assignment[role]);ids,y=valid_ids[positions],valid_y[positions];all_positions.append(positions)
            require(np.bincount(y,minlength=3).tolist()==counts,'Exact fresh role quotas')
            path=output/(role+'.npz');np.savez(path,ids=ids,y=y);role_files[role]=receipt(path)
            id_path=output/(role+'_ids.npy');np.save(id_path,ids,allow_pickle=False);role_files[role+'_ids']=receipt(id_path)
            role_hashes[role]=dict(ids=rt.fp(torch.from_numpy(ids)),y=rt.fp(torch.from_numpy(y)))
        require(np.array_equal(np.sort(np.concatenate(all_positions)),np.arange(3942,dtype=np.int64)),'Exact disjoint original VALID union')
        def numpy_state(state):return (state[0],state[1].tolist(),state[2],state[3],state[4])
        role_state=dict(seed=491003011,before=numpy_state(before),after=numpy_state(after),class_permutations=permutations,original_VALID_ids=torch.from_numpy(valid_ids.copy()),role_hashes=role_hashes)
        role_state['fingerprint']=rt.fp(role_state);role_path=output/'ROLE_RNG_AND_PERMUTATIONS.pt';torch.save(role_state,role_path)
        # Complete public support only: union transpose, remove loops, dedupe.
        edges=rt.tensor['edge_index'];support=set()
        for u,v in zip(*edges.tolist()):
            if u!=v:support.add((u,v));support.add((v,u))
        ordered=sorted(support);neighbors=[[] for _ in range(19717)]
        for u,v in ordered:neighbors[u].append(v)
        support_tensor=torch.tensor(ordered,dtype=torch.int64).T.contiguous();support_hash=rt.fp(support_tensor)
        blocks={};payload_hashes={};geometry_costs={}
        for seed in SEEDS:
            rt.guard();tick=time.monotonic();outside=rt.fp(rt.rng());private=torch.Generator(device='cpu').manual_seed(seed+491001001);state_before=private.get_state().clone()
            anchors=torch.randperm(19717,dtype=torch.long,device='cpu',generator=private)[:64].clone();state_after=private.get_state().clone()
            distance=np.full((19717,64),16,dtype=np.uint8);visits=0
            for column,anchor in enumerate(anchors.tolist()):
                distance[anchor,column]=0;queue=deque([anchor])
                while queue:
                    u=queue.popleft();depth=int(distance[u,column])
                    if depth>=15:continue
                    for v in neighbors[u]:
                        visits+=1
                        if distance[v,column]==16:distance[v,column]=depth+1;queue.append(v)
            distance_tensor=torch.from_numpy(distance);bias=distance_tensor.to(torch.float32).div(-4)
            require(anchors.unique().numel()==64 and distance_tensor.shape==(19717,64) and int(distance_tensor.max())<=16 and bool((distance_tensor[anchors,torch.arange(64)]==0).all()) and torch.isfinite(bias).all().item(),'Complete capped shortest-path geometry')
            payload=dict(seed=seed,landmark_seed=seed+491001001,anchors=anchors,distance=distance_tensor,bias=bias,generator_before=state_before,generator_after=state_after,support_fingerprint=support_hash,ordered_native_edges_fingerprint=rt.graph['original_array_fingerprints']['edge_index'])
            require(rt.fp(rt.rng())==outside,'Private landmarks/BFS changed global RNG')
            blocks[str(seed)]=payload;payload_hashes[str(seed)]=rt.fp(payload);geometry_costs[str(seed)]=dict(BFS_seconds=time.monotonic()-tick,support_arc_visits=visits,landmarks=64,diameter_cap=16,labels_or_predictions_used=False)
        capsule=dict(schema='canonical-PubMed-LM4-landmark-BFS-v1',nomination_manifest_sha256=NOMINATION_SHA,graph=rt.graph,support=support_tensor,blocks=blocks)
        digest=rt.fp(capsule);capsule['fingerprint']=digest;path=output/'CANONICAL_LANDMARK_GEOMETRY.pt';torch.save(capsule,path)
        result.update(complete=True,status='prepared_complete',train_custody=rt.train_custody,original_VALID_custody=data['validation_custody'],original_VALID_bundle=data['valid_bundle'],graph=rt.graph,role_files=role_files,role_payload_fingerprints=role_hashes,role_rng_and_permutations=receipt(role_path),role_rng_and_permutations_fingerprint=role_state['fingerprint'],role_counts=dict(select=1313,cal=1313,dev=1316),original_VALID_exact_disjoint_union=True,geometry_capsule=receipt(path),geometry_capsule_fingerprint=digest,geometry_payload_fingerprints=payload_hashes,support_fingerprint=support_hash,geometry_costs=geometry_costs,global_RNG_unchanged=True,automatic_fits_released=False)
        rt.guard()
    except BaseException as error:
        result.update(status='failed_retained',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if rt is not None:result['costs']=rt.costs()
        write(output/'CUSTODY.json',result)
    return result
