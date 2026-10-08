from pathlib import Path
import os,json,socket,subprocess,hashlib,math
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_correction_complete_WikiCS_label_reach_20261008_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
import torch
f=P/'wikics_official_acquisition_root_20261007_v1/available/wikics_split0.pt'
assert hashlib.sha256(f.read_bytes()).hexdigest()=='74868c33350221f3a38f26f156a72003f6ee943f6275fb87c205f6bc1cf775a5'
a=torch.load(f,map_location='cpu',weights_only=True)
assert set(a)=={'x','edge_index','train_ids','train_y','valid_ids','valid_y'}
edge=a['edge_index'];A=a['train_ids'];V=a['valid_ids'];N=len(a['x']);n=len(A);k=n//2
assert N==11701 and edge.shape==(2,442907) and n==580 and len(V)==5274
src,dst=edge[:,edge[0]!=edge[1]]
legal=torch.zeros(N,dtype=torch.bool);legal[A]=True
chosen=legal[src];pairs=torch.unique(src[chosen]*N+dst[chosen]);anchor_target=pairs%N
unique_degree=torch.bincount(anchor_target,minlength=N);total_degree=torch.bincount(dst,minlength=N)
anchor_records=torch.bincount(dst[chosen],minlength=N)
def summary(ids):
 d=unique_degree[ids];rat=anchor_records[ids].float()/total_degree[ids].clamp_min(1).float()
 return {'nodes':len(ids),'with_no_anchor_neighbor':int((d==0).sum()),'with_any_anchor_neighbor':int((d>0).sum()),'fraction_any_anchor_neighbor':float((d>0).float().mean()),'mean_distinct_anchor_neighbors':float(d.float().mean()),'maximum_distinct_anchor_neighbors':int(d.max()),'mean_uniform_attention_mass_to_anchors':float(rat.mean()),'uniform_mass_is_not_learned_attention':True}
def p_empty(d):
 if d>k-1:return 0.
 return math.prod((k-1-j)/(n-1-j) for j in range(d))
prob=[p_empty(int(d)) for d in unique_degree[A].tolist()]
out={'nodes':N,'TRAIN':summary(A),'development_structure_only':summary(V),'common_mask':{'n':n,'k':k,'conditional_empty_probability_formula':'choose(k-1,d)/choose(n-1,d), distinct nonself TRAIN neighbours d','expected_fraction_TRAIN_queries_with_visible_label_neighbor':1-sum(prob)/n,'conditional_inverse_inclusion_scale':(n-1)/(n-k)},'payload_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'TEST_fields_available':False,'development_truths_scored':False,'scientific_training':False,'model_predictions_opened':False,'GPU_initialized':torch.cuda.is_initialized()}
assert not out['GPU_initialized']
D.mkdir(exist_ok=False);(D/'REPORT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
