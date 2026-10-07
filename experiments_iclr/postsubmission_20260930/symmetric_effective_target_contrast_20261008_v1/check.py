from pathlib import Path
import numpy as np,hashlib,json,socket,time,resource,sys
started=time.perf_counter();P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');assert socket.gethostname()=='anogena-2-0'
source=P/'contrastive_BE_steering_continuing_research_20261007_v1/context_positive_masks.py';assert hashlib.sha256(source.read_bytes()).hexdigest()=='7615d8441a699f297314e7c838b9eb897b9e3dbfe22d1be350742e83f21405a9';sys.path.insert(0,str(source.parent));from context_positive_masks import sampled_weights
archive=P/'context_positive_TRAIN_preflight_root_20261007_v1/FROZEN_TRAIN_TARGETS.npz';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6'
with np.load(archive,allow_pickle=False) as f:
 rows=f['panel_rows'];route=sampled_weights(f['masks'],rows,'route');common=sampled_weights(f['masks'],rows,'common');permuted=sampled_weights(f['permuted_masks'],rows,'route')
S=lambda a:(a+a.transpose(0,2,1))/2
r,c,q=map(S,(route,common,permuted));N=len(rows)
assert np.allclose(r.sum((1,2)),N) and np.allclose(c.sum((1,2)),N) and np.allclose(q.sum((1,2)),N)
metric=lambda a,b:(.5*np.abs(a-b).sum((1,2))/N).tolist()
a=metric(r,c);b=metric(r,q)
result={'schema':'symmetric-effective-target-contrast-v1','CPU_only':True,'frozen_TRAIN_target_archive_only':True,'dataset_or_model_or_checkpoint_access':False,'VALID_TEST_access':False,'scientific_fits':0,'forward_calls':0,'target_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'panel_count':N,'effective_target':'(Q+Qtranspose)/2; non-row-stochastic, normalized as a complete pair distribution of total massN','symmetric_pair_TV_route_vs_common_by_route':a,'symmetric_pair_TV_route_vs_permuted_by_route':b,'mean_symmetric_pair_TV_route_vs_common':float(np.mean(a)),'mean_symmetric_pair_TV_route_vs_permuted':float(np.mean(b)),'all_effective_target_contrasts_nonzero':bool(all(v>0 for v in a+b)),'frozen_gates_changed':False,'numpy_version':np.__version__,'seconds':time.perf_counter()-started,'peak_RSS_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'interpretation':'Actual frozen controls remain distinct after the same-target symmetric-loss quotient. This is target eligibility, not useful private gradients, competence or predictive improvement.'}
D=P/'symmetric_effective_target_contrast_20261008_v1';D.mkdir(exist_ok=False);(D/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
