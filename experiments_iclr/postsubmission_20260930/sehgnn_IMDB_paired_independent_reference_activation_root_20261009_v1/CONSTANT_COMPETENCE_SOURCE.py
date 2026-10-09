from pathlib import Path
import json,math,hashlib,importlib.util,sys,subprocess,socket,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_IMDB_paired_independent_reference_activation_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
cfg=json.loads((A/'RELEASE.json').read_text());terminal=json.loads((A/'TERMINAL.json').read_text());assert terminal['exit_code']==0 and terminal['child_reaped']
assert not Path('/proc/578224').exists() and not Path('/proc/578226').exists()
report=json.loads((Path(cfg['entry_output_directory'])/'ROOT_REPORT.json').read_text());assert report['complete'] and report['completed_actual_body_fits']==24
source=P/'imdb_role_isolated_public_schema_backbone_preparation_20261009_v1/role_loader.py';spec=importlib.util.spec_from_file_location('_competence_public_label_parser',source);loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader);loader.source_gate()
label=Path(cfg['input_root'])/'label.dat';assert hashlib.sha256(label.read_bytes()).hexdigest()==cfg['expected_input_files']['label.dat']['sha256']
labels=loader.development_labels(label,4932,5)
def f1(truth,prediction):
 t=sum(y and p for row in truth for y,p in zip(row,prediction));fp=sum((not y) and p for row in truth for y,p in zip(row,prediction));fn=sum(y and (not p) for row in truth for y,p in zip(row,prediction));return 2*t/(2*t+fp+fn) if 2*t+fp+fn else 0.
rows=[]
for seed in (1,2,3):
 role=json.loads(Path(cfg['roles'][str(seed)]['path']).read_text());train,valid=loader.check_role(role,labels,5,cfg['expected_input_files']);prior=[sum(labels[node][j] for node in train)/len(train) for j in range(5)];truth=[labels[node] for node in valid];assert all(0<p<1 for p in prior)
 bce=-sum(math.log(p if y else 1-p) for row in truth for y,p in zip(row,prior))/(len(truth)*5)
 rows.append(dict(role_seed=seed,train_prior=prior,VALID_rows=len(truth),train_prevalence_constant_micro_F1=f1(truth,[p>.5 for p in prior]),all_positive_constant_micro_F1=f1(truth,[True]*5),TRAIN_prior_constant_VALID_BCE=bce))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,constant_statistics_only=True,new_fit_or_model_forward_count=0,VALID_parameter_selection=False,TEST_access=False,development_label_sha256=cfg['expected_input_files']['label.dat']['sha256'])))
