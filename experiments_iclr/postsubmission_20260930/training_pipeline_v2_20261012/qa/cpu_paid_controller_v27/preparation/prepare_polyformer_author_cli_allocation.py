"""Stage unchanged PolyFormer CLI bytes/raw data and import-only qualification.

This program never starts an author training process or creates a GO receipt.
The eight official commands remain disabled until root integrates the gates.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import zlib

from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PHASE = PACKAGE.parent
SOURCE = PHASE / "coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification"
LOCAL = HERE / "polyformer_author_cli_allocation_20261012_v1"
DESTINATION = "massive_polyformer_author_cli_packet_20261012_v1"
DATASETS = {"citeseer": "citeseer", "pubmed": "pubmed", "cs": "coauthor-cs", "physics": "coauthor-physics"}
PROVIDER_NAMES = {"citeseer": "citeseer", "pubmed": "pubmed", "cs": "CS", "physics": "Physics"}
SOURCE_FILES = ["training.py", "node_classification.sh", "mymodels.py", "utils.py", "layers/PolyFormerBlock.py", "dataloader/dataset_loader.py"]
SPLIT_SEEDS = [1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def contract():
    files = []
    for name in SOURCE_FILES:
        raw = (SOURCE / name).read_bytes()
        files.append({"path": name, "bytes": len(raw), "sha256": sha(raw), "data": base64.b64encode(raw).decode()})
    source_text = (SOURCE / "training.py").read_text()
    ast.parse(source_text)
    assert "for RP in range(args.runs):" in source_text
    assert "default=250" in source_text and "default=2000" in source_text
    assert "if val_acc > best_val_acc" in source_text
    assert "best_val_acc = test_acc = 0" in source_text
    assert "SEEDS=" + str(SPLIT_SEEDS).replace(" ", "") in source_text
    jobs, excluded = [], []
    for line in (SOURCE / "node_classification.sh").read_text().splitlines():
        if not line.startswith("python "):
            continue
        argv = shlex.split(line)
        dataset = argv[argv.index("--dataset") + 1]
        base = argv[argv.index("--base") + 1]
        if dataset not in DATASETS:
            excluded.append({"dataset": dataset, "base": base, "reason": "no_materialized_raw_provider_custody_in_this_packet"})
            continue
        identifier = dataset + "-" + base
        jobs.append({"job_id": identifier, "source_order": len(jobs), "dataset": dataset,
                     "canonical_dataset": DATASETS[dataset], "base": base,
                     "original_script_argv": argv, "python_argv": argv[1:],
                     "cwd": "workdirs/" + identifier, "runs_in_one_process": 10,
                     "output": {"stdout": "quarantine/" + identifier + ".stdout.log",
                                "stderr": "quarantine/" + identifier + ".stderr.log"},
                     "provider_raw_dir": "datasets/" + PROVIDER_NAMES[dataset] + "/raw",
                     "launch_enabled": False})
    assert len(jobs) == 8 and len(excluded) == 12
    manifests = {}
    mirror = HERE / "polyformer_allocation_20261012_v1/allocation_artifacts/massive_polyformer_packet_20261012_v1/data"
    for dataset in DATASETS.values():
        path = mirror / dataset / "CUSTODY.json"
        value = json.loads(path.read_text())
        raw = [row for row in value["provider_files"] if row["kind"] == "provider_raw_file"]
        provider = [row for row in value["provider_files"] if row["kind"] == "installed_provider_source"]
        assert raw and len(provider) == 1
        manifests[dataset] = {"custody_sha256": sha(path.read_bytes()), "provider_sources": provider, "raw_files": raw}
    result = {
        "schema": "polyformer-unchanged-author-cli-contract-v1", "destination": DESTINATION,
        "source_repository": "https://github.com/air029/PolyFormer",
        "source_commit": "d390f39e88d0eaac80318fdc7704bd3cf8b13",
        "source_script_sha256": next(r["sha256"] for r in files if r["path"] == "node_classification.sh"),
        "roster": jobs, "excluded_source_commands": excluded,
        "planned_cli_processes": 8, "planned_model_fits": 80,
        "maximum_updates_per_fit": 2000, "maximum_total_optimizer_updates": 160000,
        "maximum_train_model_forwards": 160000, "maximum_validation_test_model_forwards": 640000,
        "actual_author_training_processes_started": 0, "actual_author_model_fits": 0,
        "protocol": {
            "global_rng_seed": 42, "global_rng_reset": "unchanged mymodels import and unchanged main; once per CLI process",
            "split_seeds_in_source_order": SPLIT_SEEDS,
            "run_order": "RP0 through RP9 sequentially in one process; never split/reset individual repeats",
            "constructor_before_split": True, "constructor_and_dropout_rng_advance_between_repeats": True,
            "split_rule": "percls_trn=int(round(.6*N/classes)); val_lb=int(round(.2*N)); NumPy RandomState(split_seed); class-balanced TRAIN, all class examples if smaller; random VAL from remainder; TEST rest",
            "preprocessing": "unchanged DataLoader NormalizeFeatures; undirected conversion if needed; native mono/cheb basis computed in fresh empty private bases directory",
            "optimizer": "unchanged Adam; one parameter group per registered parameter; attnmodule LR/decay versus source ordinary LR/decay",
            "stopping": "maximum2000; bad counter reset only on strict VAL accuracy improvement above initial0; stop at counter==250",
            "selector": "first strict VAL accuracy maximum; retains corresponding TEST accuracy; no saved checkpoints",
            "per_epoch_TEST": "unchanged TEST accuracy/loss diagnostics remain inside quarantined source output; not used in stopping or optimization",
            "evaluation_forward_count": "source test() computes model(data) once for accuracy and three more times for TRAIN/VAL/TEST losses: four eval forwards per epoch",
            "reporting": "unchanged ten-repeat mean/variance and seaborn1000-bootstrap95% interval; bootstrap seed is not supplied by source",
            "calibration": "none in original CLI track", "published_environment_match": False,
        },
        "gates": {
            "launch_enabled": False, "GO_created": False,
            "required_before_any_CLI_process": ["explicit human GO bound to final master roster", "qualified GPU/resource placement", "whole master HPO/selection roster closure recorded by root", "parent integrates source/roster-bound execution gate"],
            "quarantine_release": "only after all eight admitted CLI processes reach declared terminal status and complete author roster closure is frozen; preserve every failure and cost",
            "monitor_policy": "read process/status/cost metadata only; no reading, parsing, tailing or surfacing epoch output or partial author aggregates before whole-roster closure",
            "outcome_adaptation": "none; no outcome-driven settings, repeat deletion, roster expansion or rerun",
        },
        "runtime": {
            "python": "repository .venv/bin/python", "published_recommendation": {"torch": "2.0.0", "torch_geometric": "2.3.0"},
            "allocation_departures": {"torch": "2.1.2+cu118", "torch_geometric": "2.7.0", "dgl": "2.4.0+cu118", "seaborn": "0.13.2 private overlay; upstream repository does not pin version"},
            "source_bytes_unchanged": True, "environment_bitwise_reproduction_claim": False,
            "private_environment": {"PYTHONPATH": "dependencies/site", "LD_LIBRARY_PATH": "existing repository .venv torch/lib plus nvidia/*/lib; child only", "DGLBACKEND": "pytorch"},
        },
        "raw_custody": manifests,
        "scientific_launch_enabled": False,
    }
    return result, files


REMOTE = GUARD + r'''
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
stage=phase/payload['contract']['destination'];assert not stage.exists();stage.mkdir()
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(stage)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
source=stage/'source';source.mkdir()
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[{k:v for k,v in row.items() if k!='data'} for row in payload['files']])
atomic(stage/'AUTHOR_ROSTER.json',payload['contract'])
corpus=stage/'raw_provider_corpus';corpus.mkdir();ancestry=[]
for dataset,value in payload['contract']['raw_custody'].items():
 for row in value['provider_sources']:
  original=Path(row['path']);assert original.is_relative_to(repo)
  assert hashlib.sha256(original.read_bytes()).hexdigest()==row['sha256']
 folder=corpus/dataset;folder.mkdir()
 for row in value['raw_files']:
  original=Path(row['path']);assert original.is_relative_to(phase)
  assert not original.is_symlink();raw=original.read_bytes()
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  target=folder/original.name;target.write_bytes(raw);target.chmod(0o444)
  ancestry.append({'source':row,'copy':receipt(target),'original_modified':False,'target_tensor_decoded':False})
atomic(stage/'RAW_CACHE_ANCESTRY.json',ancestry)
workdirs=stage/'workdirs';workdirs.mkdir();(stage/'quarantine').mkdir()
for job in payload['contract']['roster']:
 work=stage/job['cwd'];work.mkdir()
 for row in payload['files']:
  target=work/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/row['path'],target);target.chmod(0o444)
 raw_dir=work/job['provider_raw_dir'];raw_dir.mkdir(parents=True)
 for raw_file in (corpus/job['canonical_dataset']).iterdir():shutil.copyfile(raw_file,raw_dir/raw_file.name);(raw_dir/raw_file.name).chmod(0o444)
 (work/'bases').mkdir();(work/'datasets'/raw_dir.parent.name/'processed').mkdir()
dependencies=stage/'dependencies';dependencies.mkdir();wheels=dependencies/'wheels';wheels.mkdir();private=dependencies/'site'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',DGLBACKEND='pytorch')
tmp=stage/'engineering_tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
site=repo/'.venv/lib/python3.11/site-packages';libdirs=[site/'torch/lib']+sorted((site/'nvidia').glob('*/lib'))
env['LD_LIBRARY_PATH']=':'.join(map(str,libdirs))+':'+env.get('LD_LIBRARY_PATH','')
python=str(repo/'.venv/bin/python')
def run(argv,label,cwd,timeout=60):
 tick=time.monotonic()
 try:
  child=subprocess.run(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
  result={'exit_code':child.returncode,'stdout':child.stdout,'stderr':child.stderr}
 except subprocess.TimeoutExpired as error:
  result={'exit_code':None,'timeout':True,'stdout':str(error.stdout),'stderr':str(error.stderr)}
 result.update({'command':argv,'wall_seconds':time.monotonic()-tick,'engineering_import_only':True,'scientific_fits':0,'benchmark_scores':0})
 atomic(stage/(label+'.json'),result);return result
download=run([python,'-B','-m','pip','download','--no-deps','--only-binary=:all:','--dest',str(wheels),'seaborn==0.13.2'],'SEABORN_WHEEL_DOWNLOAD',stage,90)
install={'exit_code':None}
if download['exit_code']==0:
 wheel_list=list(wheels.glob('seaborn-0.13.2-*.whl'));assert len(wheel_list)==1
 atomic(stage/'SEABORN_WHEEL_INVENTORY.json',[receipt(p) for p in wheel_list])
 install=run([python,'-B','-m','pip','install','--no-deps','--no-index','--target',str(private),str(wheel_list[0])],'SEABORN_PRIVATE_INSTALL',stage,60)
env['PYTHONPATH']=str(private)
checks=[]
if install['exit_code']==0:
 code="import importlib.metadata as m,json;import dgl,seaborn as sns;assert callable(sns.utils.ci) and callable(sns.algorithms.bootstrap);print(json.dumps({n:m.version(n) for n in ['torch','torch-geometric','dgl','seaborn','numpy','scipy','pandas','matplotlib','networkx','ogb','torch-scatter','torch-sparse']}))"
 checks.append(run([python,'-B','-c',code],'AUTHOR_IMPORT_AND_RUNTIME_CHECK',stage,60))
 if checks[-1]['exit_code']==0:
  for job in payload['contract']['roster']:
   checks.append(run([python,'-B','-u','training.py','--help'],'CLI_HELP_'+job['job_id'],stage/job['cwd'],60))
postcheck=[]
for job in payload['contract']['roster']:
 work=stage/job['cwd'];assert list((work/'bases').iterdir())==[]
 assert list((work/'datasets'/Path(job['provider_raw_dir']).parent.name/'processed').iterdir())==[]
 for row in payload['files']:assert hashlib.sha256((work/row['path']).read_bytes()).hexdigest()==row['sha256']
 postcheck.append({'job_id':job['job_id'],'source_bytes_unchanged':True,'bases_empty':True,'processed_empty':True})
assert list((stage/'quarantine').iterdir())==[]
atomic(stage/'POST_QUALIFICATION_BYTE_AND_EMPTY_DIR_CHECK.json',postcheck)
ready=install['exit_code']==0 and len(checks)==9 and all(r['exit_code']==0 for r in checks)
summary={'schema':'polyformer-author-cli-preparation-v1','destination':str(stage.relative_to(phase)),
 'contract_sha256':receipt(stage/'AUTHOR_ROSTER.json')['sha256'],'source_files':len(payload['files']),
 'raw_provider_files':len(ancestry),'planned_cli_processes':8,'planned_model_fits':80,
 'dependency_and_unchanged_cli_help_qualified':ready,'source_bytes_unchanged':True,
 'original_raw_cache_modified':False,'heldout_target_tensor_decoded':False,
 'model_or_optimizer_constructed':False,'scientific_training_processes_started':0,'benchmark_scores':0,
 'GO_created':False,'scientific_launch_enabled':False,
 'remaining_launch_requirements':payload['contract']['gates']['required_before_any_CLI_process'],
 'published_environment_exact_match':False,
 'blocked_dependencies':[] if ready else ['inspect_retained_import_help_and_dependency_receipts']}
atomic(stage/'PREPARATION_SUMMARY.json',summary)
for p in [stage/'SOURCE_INVENTORY.json',stage/'AUTHOR_ROSTER.json',stage/'RAW_CACHE_ANCESTRY.json']:p.chmod(0o444)
print(json.dumps({'summary':summary,'checks':[{'exit_code':r['exit_code'],'command':r['command']} for r in checks]}))
'''


def main():
    LOCAL.mkdir(exist_ok=False)
    value, files = contract()
    (LOCAL / "AUTHOR_ROSTER_DRAFT.json").write_text(json.dumps(value, indent=2) + "\n")
    (LOCAL / "SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in r.items() if k != "data"} for r in files], indent=2) + "\n")
    encoded = base64.b64encode(zlib.compress(json.dumps({"contract":value,"files":files}).encode(),9)).decode()
    result = invoke(REMOTE, [encoded], LOCAL, "PREPARATION_TRANSPORT", 600)
    print(json.dumps(result))
    raise SystemExit(0 if result["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
