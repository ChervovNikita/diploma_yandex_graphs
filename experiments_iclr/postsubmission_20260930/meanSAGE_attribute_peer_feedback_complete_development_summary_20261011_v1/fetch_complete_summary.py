"""Fetch only a successfully closed complete peer family; no model/reader run."""
import base64,datetime,hashlib,importlib.util,json,pathlib,shlex,zlib
HERE=pathlib.Path(__file__).resolve().parent;P=HERE.parent
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs';UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
remote=r"""
import base64,datetime,hashlib,json,math,pathlib,socket,subprocess,sys,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';E=P/'meanSAGE_attribute_peer_feedback_acquisition_root_20261011_v1';W=P/'meanSAGE_attribute_peer_feedback_readout_after_whole_acquisition_root_20261011_v1'/'watch_v1'
BLOCKS=(10103,10427,10949);ARMS=('P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4');EXPECTED={(b,a) for b in BLOCKS for a in ARMS}
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def closed(p):
 a=json.loads(p.read_text());assert a['status']=='COMPLETE_EXIT_0' and a['exit_code']==0 and a['direct_child_wait'] and a['child_pid_absent'] and a['owned_cuda_pid_absent'] and not a['timed_out'];return a
# Read lifecycle/closure metadata first; no report or scalar-fit quality before admission.
watch_path=W/'WATCH_END.json';watch=json.loads(watch_path.read_text())
assert watch['complete'] and watch['all30banks_48states_57bodies_before30CAL_then_onewholeDEV'] and watch['all_CAL_owners_closed_before_DEV'] and not watch['partial_metrics_opened'] and not watch['model_retry'] and not watch['readout_retry']
owners={str(p.relative_to(P)):dict(receipt=receipt(p),value=closed(p)) for p in [E/'owner_v1/OWNER_END.json',W/'calibrate_owner_v1/OWNER_END.json',W/'score_owner_v1/OWNER_END.json']}
whole_path=W/'WHOLE_FAMILY_CLOSURE.json';whole=json.loads(whole_path.read_text());assert whole['complete'] and (whole['banks'],whole['optimizer_trajectories'],whole['physical_backbones'],whole['exports'])==(30,48,57,30)
calclose_path=W/'ALL_CAL_CLOSURE.json';calclose=json.loads(calclose_path.read_text());assert calclose['complete'] and calclose['endpoints']==30 and calclose['scalar_updates']==15000 and calclose['CAL_owner_waited_and_reaped']
cp=W/'calibrate_v1/TEMPERATURES.json';ct=json.loads(cp.read_text());assert ct['complete'] and ct['endpoints']==30 and ct['actual_scalar_updates']==15000 and ct['DEV_truth_read'] is False and len(ct['records'])==30 and {(r['block'],r['arm']) for r in ct['records']}==EXPECTED
assert all(r['status']=='fixed_endpoint_complete' and r['updates']==500 and r['endpoint']=='latest_tied_minimum_step0_to500' and r['operator']=='mean_m softmax(member_log_probs_m/T)' and 0<=r['selected_step']<=500 and math.isfinite(r['logT']) and -5<=r['logT']<=5 and r['temperature']==math.exp(r['logT']) for r in ct['records'])
assert receipt(cp)==calclose['temperatures']
pp=E/'acquisition_v1/COMPLETE_PANEL.json';panel=json.loads(pp.read_text());assert panel['complete'] and panel['terminal'] and (panel['banks'],panel['optimizer_trajectories'],panel['physical_backbones'],panel['exports'],panel['selected_optimizer_states'],panel['selected_physical_bodies'])==(30,48,57,30,48,57)
assert panel['CAL_DEV_truth_read'] is False and panel['serving']=='mean_class_probabilities' and len(panel['records'])==30 and {(r['block'],r['arm']) for r in panel['records']}==EXPECTED
assert sum(len(r['fits']) for r in panel['records'])==48 and sum(r['physical_backbones'] for r in panel['records'])==57 and all(f['status']=='selected_complete' and f['selected_restore_verified'] for r in panel['records'] for f in r['fits'])
assert receipt(pp)==whole['panel']
# Whole scientific closure is established. Now retain every arm and contrast.
rp=W/'score_v1/DEVELOPMENT_REPORT.json';bound=receipt(rp);assert bound==watch['report'];a=json.loads(rp.read_text())
assert a['complete'] and not a['retry'] and not a['survivor_subset'] and len(a['records'])==30 and len(a['comparisons'])==90 and a['calibration_scalar_updates']==15000
assert {(r['block'],r['arm']) for r in a['records']}==EXPECTED and {(r['candidate'],r['reference']) for r in a['comparisons']}=={(c,r) for c in ARMS for r in ARMS if c!=r}
release=a['release'];assert release['one_whole_release'] and release['blocks']==list(BLOCKS) and (release['after_optimizer_trajectories'],release['after_physical_bodies'],release['after_exports'],release['after_CAL_endpoints'])==(48,57,30,30)
assert a['panel']==receipt(pp) and a['temperatures']==receipt(cp)
summary={k:v for k,v in a.items() if k not in ('records','comparisons')};summary['records']=[{k:v for k,v in r.items() if k!='classes'} for r in a['records']]
summary['comparisons']=[{**{k:v for k,v in r.items() if k!='per_block'},'per_block':[{k:v for k,v in d.items() if k!='classes'} for d in r['per_block']]} for r in a['comparisons']]
summary.update(full_report_receipt=bound,all_per_class_results_retained_in_full_remote_report=True,full_checkpoints_logits_and_targets_remain_remote=True,no_new_fit_forward_reader_or_calibration=True)
calrows=[{k:v for k,v in r.items() if k!='CAL_minimum_NLL'} for r in ct['records']]
summary['fixed_CAL_metadata']=dict(complete=True,endpoints=30,scalar_updates=15000,seconds=ct['seconds'],receipt=receipt(cp),fitted_CAL_quality_opened_for_interpretation=False,records=calrows)
summary['acquisition_custody_metadata']=dict(receipt=receipt(pp),counts={k:panel[k] for k in ['banks','optimizer_trajectories','physical_backbones','exports','selected_optimizer_states','selected_physical_bodies']},graph_metadata=panel['graph'],selected_fit_metadata=panel['records'])
summary['owners']=owners;summary['watch']=dict(receipt=receipt(watch_path),value=watch);summary['closure_receipts']=dict(whole_family=dict(receipt=receipt(whole_path),value=whole),all_CAL=dict(receipt=receipt(calclose_path),value=calclose))
summary['source_manifest']=whole['source_manifest'];assert summary['source_manifest']['sha256']=='f83cb34bae29bda7bd542075bb3009d973fc7ab3bf494bd2cdfcf0801e704cd4'
data=(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode();print(json.dumps(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())))
"""
spec=importlib.util.spec_from_file_location('peer_transport_v19',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cmd=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,R,UUID,'complete_summary'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',cmd]
assert not (HERE/'SUMMARY.json').exists(),'Existing outcome retained; do not refetch over it'
name=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
try:r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=45)
except BaseException as error:
 record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),transport_error=dict(type=type(error).__name__,message=str(error)),scientific_failure_inferred=False)
 with (HERE/('FETCH_FAILURE_'+name+'.json')).open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
 raise
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr)
if r.returncode:
 record['stdout']=r.stdout
 with (HERE/('FETCH_FAILURE_'+name+'.json')).open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
 print(json.dumps(dict(exit_code=r.returncode,fetch_admission_failed=True,scientific_failure_inferred=False)));raise SystemExit(r.returncode)
row=json.loads(r.stdout);data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];a=json.loads(data)
with (HERE/'SUMMARY.json').open('xb') as f:f.write(data)
record.update(bytes=row['bytes'],sha256=row['sha256'])
with (HERE/'FETCH_TRANSPORT.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(complete=a['complete'],banks=len(a['records']),contrasts=len(a['comparisons']),scalar_updates=a['calibration_scalar_updates'],full_report=a['full_report_receipt'],source=a['source_manifest'],summary_receipt=dict(path=str(HERE/'SUMMARY.json'),bytes=len(data),sha256=row['sha256']),complete_screen_pass=a['gates']['complete_development_screen_pass']),indent=2))
