from pathlib import Path
import json,hashlib,socket,time,resource
import numpy as np
import torch
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_post_family_prediction_collection_root_20261008_v1';O=P/'label_only_correction_margins_retrospective_root_20261008_v1'
assert socket.gethostname()=='anogena-2-0'
assert torch.__version__=='2.1.2+cu118';torch.set_num_threads(1)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
c=json.loads((D/'COMPLETE.json').read_text());assert c['complete'] and c['work']['raw_files_completed']==12
opening=P/'label_only_four_bank_postfamily_activation_root_20261008_v1/WHOLE_FAMILY_OPENING.json';assert json.loads(opening.read_text())['root_comparative_opening_authorized']
role=P/'label_correction_full_WikiCS_role_projection_allocation_20261008_v1/valid.npz';assert sha(role)=='99114cb1d50b876e9cdb2d4a1c116e217f0a8aa2583a2561d0491461c0c0ec48'
with np.load(role,allow_pickle=False) as data:truth=torch.from_numpy(data['y'].copy()).long();ids=torch.from_numpy(data['ids'].copy()).long()
def summary(x):
 if not x.numel():return None
 x=x.double().reshape(-1);return dict(count=x.numel(),mean=float(x.mean()),median=float(x.median()),p90=float(torch.quantile(x,.9)),p99=float(torch.quantile(x,.99)),maximum=float(x.max()))
started=time.monotonic();rows=[]
for b in c['raw_predictions']:
 f=D/b['path'];assert sha(f)==b['sha256'];v=torch.load(f,map_location='cpu',weights_only=False);assert torch.equal(v['valid_ids'],ids)
 base=v['native_logits'];bank=v['member_logits'];prob=v['native_probabilities'];pred=base.argmax(-1);ok=pred==truth
 delta=bank-base[None];centered=delta-delta.mean(-1,keepdim=True)
 other=base.clone();other.scatter_(1,truth[:,None],float('-inf'));gap=other.max(-1).values-base.gather(-1,truth[:,None]).squeeze(-1)
 uplift=delta.gather(-1,truth[None,:,None].expand(len(bank),-1,1)).squeeze(-1)-delta.gather(-1,pred[None,:,None].expand(len(bank),-1,1)).squeeze(-1)
 row=dict(seed=v['seed'],arm=v['arm'],selected_epoch=v['selected_epoch'],raw_sha256=b['sha256'],cohorts={})
 for name,mask in [('whole',torch.ones_like(ok)),('own_native_wrong',~ok),('own_native_correct',ok)]:
  row['cohorts'][name]=dict(support=int(mask.sum()),native_top1_confidence=summary(prob[mask].max(-1).values),native_true_class_shortfall=summary(gap[mask]),centered_residual_Linf=summary(centered[:,mask].abs().max(-1).values),true_vs_native_rival_uplift=summary(uplift[:,mask]),nonzero_residual_nodes=int((centered[:,mask].abs().max(-1).values.max(0).values>0).sum()),candidate_member_correct_nodes=int((bank[:,mask].argmax(-1)==truth[mask][None]).any(0).sum()))
 rows.append(row)
O.mkdir(exist_ok=False);out=dict(schema='postfamily-retrospective-selected-correction-margin-diagnostics-v1',full_family_complete=True,raw_outcomes_opened_after_full_family=True,complete_collector_sha256=sha(D/'COMPLETE.json'),qualification_or_new_fit=False,reselection=False,unseen_confirmation=False,cohorts='Whole and errors/correct decisions of each arm own selected-state feature-only base. Descriptive, not gate subsets.',rows=rows,wall_seconds=time.monotonic()-started,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(O/'DIAGNOSTICS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,allow_nan=False))
