from pathlib import Path
import socket,subprocess,json,os,hashlib,statistics,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_IMDB_literal_five_seed_native_reference_activation_root_20261009_v2';D=P/'sehgnn_IMDB_literal_five_seed_native_reference_execution_root_20261009_v2';O=P/'sehgnn_native5_complete_quality_readout_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
read=lambda f:json.loads(f.read_text())
bind=lambda f:dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
l,t=read(A/'LAUNCH.json'),read(A/'TERMINAL.json');assert t['exit_code']==0 and t['child_reaped'] and t['original_pid_absent'] and t['stop_reason'] is None
for h in [l['parent'],l['child']]:
 assert not Path('/proc',str(h['pid'])).exists()
 try:os.killpg(h['group'],0)
 except ProcessLookupError:pass
 else:raise ValueError('Original native group still present; no quality opening')
active={int(v.strip()) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if v.strip().isdigit()};assert not active.intersection(h['pid'] for h in [l['parent'],l['child']])
r=read(D/'COHORT_REPORT.json');assert r['status']=='complete' and r['complete'] and r['mode']=='native_reference' and r['seeds']==[1,2,3,4,5]
assert len(r['fits'])==5 and {v['seed'] for v in r['fits']}=={1,2,3,4,5};fits=[];bindings=[bind(A/'LAUNCH.json'),bind(A/'TERMINAL.json'),bind(D/'COHORT_REPORT.json')]
for seed in [1,2,3,4,5]:
 f=D/('seed'+str(seed))/'RESULT.json';v=read(f);assert v==next(x for x in r['fits'] if x['seed']==seed)
 assert v['complete'] and v['status']=='complete' and v['fresh_selected']['model_optimizer_scaler_and_streams_exact'] and v['master_scaler_state_preserved']
 assert v['native_parameter_count']==83659532 and not v['TEST_truth'];fits.append(v);bindings.append(bind(f))
summary={}
for role in ['TRAIN','VALID']:
 summary[role]={}
 for metric in ['BCE','served_micro_F1','served_macro_F1']:
  values=[v['fresh_selected']['fresh_scores'][role][metric] for v in fits];mean=statistics.mean(values);sd=statistics.stdev(values);rad=2.7764451051977987*sd/(5**.5);summary[role][metric]=dict(per_seed=values,mean=mean,sample_SD=sd,descriptive_seed_split95_interval=[mean-rad,mean+rad])
report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_five_complete=True,source='fresh selected-state serving',summary=summary,complete_fits=fits,inputs=bindings,actual_owned_process_group_CUDA_absence_verified=True,terminal=t,scientific_scope='Five predeclared optimizer/split seeds on one graph, all chosen by native development BCE. Intervals are descriptive; no independent graph generalization or unused truth claim.',GNNM_comparison_performed=False,genuine_independent4_established=False,TEST_truth_accessed=False,checkpoint_or_prediction_arrays_downloaded=False)
O.mkdir(exist_ok=False);(O/'READOUT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+chr(10))
print(json.dumps({'summary':summary,'seeds':[{'seed':v['seed'],'epochs':v['counters']['epochs_completed'],'selected_epoch':v['fresh_selected']['epoch'],'seconds':v['seconds']} for v in fits],'complete_panel_seconds':t['seconds'],'readout_path':str(O)}))
