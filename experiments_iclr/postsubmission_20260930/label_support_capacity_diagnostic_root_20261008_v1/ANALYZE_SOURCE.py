from pathlib import Path
import json,socket,hashlib,time,resource
import torch,numpy as np
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');assert socket.gethostname()=='anogena-2-0';torch.set_num_threads(1);started=time.monotonic()
owner=json.loads((P/'label_only_four_bank_WikiCS_scientific_owner_source_20261008_v2/SOURCE_BINDINGS.json').read_text())
for row in owner['roles'].values():assert hashlib.sha256((P/row['path']).read_bytes()).hexdigest()==row['sha256']
with np.load(P/owner['roles']['train']['path'],allow_pickle=False) as f:edges,train_ids,train_y=[torch.from_numpy(f[k].copy()) for k in ('edge_index','ids','y')]
with np.load(P/owner['roles']['valid']['path'],allow_pickle=False) as f:ids,truth=[torch.from_numpy(f[k].copy()) for k in ('ids','y')]
classes=torch.full((11701,),-1,dtype=torch.long);classes[train_ids]=train_y
keep=(edges[0]!=edges[1])&(classes[edges[0]]>=0);s,t=edges[:,keep]
hist=torch.zeros((11701,10),dtype=torch.long);hist.index_put_((t,classes[s]),torch.ones(len(s),dtype=torch.long),accumulate=True);h=hist[ids];distinct=(h>0).sum(-1);rows=[]
for k in range(11):
 mask=distinct==k
 if mask.any():rows.append(dict(distinct_visible_label_classes=k,support=int(mask.sum()),true_class_present_count=int((h.gather(1,truth[:,None])[:,0]>0)[mask].sum()),label_neighbor_records_sum=int(h[mask].sum())))
collector=P/'label_only_four_bank_post_family_prediction_collection_root_20261008_v1';done=json.loads((collector/'COMPLETE.json').read_text());assert done['complete'];byseed=[]
for rec in done['raw_predictions']:
 if rec['arm']!='C4':continue
 f=collector/rec['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];r=torch.load(f,map_location='cpu',weights_only=False);assert torch.equal(r['valid_ids'],ids);b=r['native_probabilities'].argmax(-1);c=r['served_probabilities'].argmax(-1);mist=b!=truth;stats=[]
 for k in range(11):
  m=distinct==k
  if m.any():stats.append(dict(distinct_visible_label_classes=k,support=int(m.sum()),own_native_errors=int((m&mist).sum()),native_error_true_class_present=int((m&mist&(h.gather(1,truth[:,None])[:,0]>0)).sum()),repairs=int((m&mist&(c==truth)).sum()),harms=int((m&~mist&(c!=truth)).sum())))
 byseed.append(dict(seed=rec['seed'],native_selected_epoch=r['selected_epoch'],rows=stats))
O=P/'label_support_capacity_diagnostic_root_20261008_v1';O.mkdir(exist_ok=False);out=dict(whole_population=5274,structural_rows=rows,C4_same_native_error_panels=byseed,truths_used_only_for_retrospective_diagnostic=True,TEST_access=False,new_fits_or_native_forwards=0,subset_gate_or_new_selection=False,raw_arrays_not_saved=True,wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(O/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
