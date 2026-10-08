from pathlib import Path
import socket,subprocess,hashlib,importlib.util,json,numpy as np,time
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
archive=P/'context_positive_TRAIN_preflight_root_20261007_v1/FROZEN_TRAIN_TARGETS.npz';source=P/'contrastive_BE_steering_continuing_research_20261007_v1/context_positive_masks.py'
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert h(archive)=='80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6' and h(source)=='7615d8441a699f297314e7c838b9eb897b9e3dbfe22d1be350742e83f21405a9'
started=time.monotonic();spec=importlib.util.spec_from_file_location('fixed_context_masks',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with np.load(archive,allow_pickle=False) as f:
 assert set(f.files)=={'train_ids','train_labels','panel_rows','masks','permuted_masks','permutations'}
 masks=f['masks'];permuted=f['permuted_masks'];panel=f['panel_rows']
route=m.sampled_weights(masks,panel,'route');common=m.sampled_weights(masks,panel,'common');shuffle=m.sampled_weights(permuted,panel,'route')
sym=lambda x:.5*(x+x.transpose(0,2,1));halfL1=lambda x:(.5*np.abs(x).sum(-1).mean(-1)).tolist()
a=route-common;b=route-shuffle
result=dict(schema='TRAIN-only-fixed-target-operative-symmetry-readout-v1',target_archive_sha256=h(archive),mask_source_sha256=h(source),fixed_panel_rows=512,
 directed_route_minus_common_per_route_halfL1=halfL1(a),symmetric_route_minus_common_per_route_halfL1=halfL1(sym(a)),
 directed_route_minus_permuted_per_route_halfL1=halfL1(b),symmetric_route_minus_permuted_per_route_halfL1=halfL1(sym(b)),
 all_four_route_common_symmetric_operators_differ=bool(np.all(np.max(np.abs(sym(a)),axis=(1,2))>0)),all_four_route_permuted_symmetric_operators_differ=bool(np.all(np.max(np.abs(sym(b)),axis=(1,2))>0)),
 interpretation='Symmetric matrices are score-gradient coefficients, not row probabilities. This is operator difference, not a private-parameter-gradient, noise, quality or novelty measurement.',
 TRAIN_only=True,VALID_TEST_or_pending_outcomes_accessed=False,new_fits=0,new_model_calls=0,active_protocol_gates_or_targets_changed=False,seconds=time.monotonic()-started)
out=P/'context_target_geometry_readout_root_20261008_v1';out.mkdir(exist_ok=False);(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
