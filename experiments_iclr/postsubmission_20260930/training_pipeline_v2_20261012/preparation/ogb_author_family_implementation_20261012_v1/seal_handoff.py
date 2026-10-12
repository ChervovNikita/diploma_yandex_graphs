"""Seal metadata handoff without launching or importing numerical code."""
import base64
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

P=Path('/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/training_pipeline_v2_20261012')
O=P/'preparation/ogb_author_family_implementation_20261012_v1'
owned=['ogb_author_cli_spec.py','run_ogb_author_cli.py','ogb_declared_seed_entry.py','compile_ogb_author_study.py','tests/test_ogb_author_cli.py','author_cli_spec.py','run_author_cli.py']
def receipt(path):
 b=path.read_bytes();return {'path':str(path.relative_to(P)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
source=[receipt(P/name) for name in owned]
seed_history={'schema':'ogb-prospective-controlled-seed-policy-revision-v1','date_UTC':datetime.now(timezone.utc).isoformat(),
 'earlier_root_policy_discussion':'701xx seed family discussed prospectively; exact earlier list is not restated or inferred here.',
 'earlier_policy_human_mandated':False,'earlier_policy_executed':False,
 'current_packet':'ogb_batchnorm_author_cli_packet_20261012_v1',
 'accepted_fresh_seed_ids':[73101,73103,73107,73109,73121,73127,73133,73141,73147,73151],
 'revision_authority':'Root prospectively accepted the fresh731xx list in coordination on12October2026; no human scientific GO.',
 'policy':'Ten independently seeded processes per frozen recipe; external wrapper seeds Python/NumPy/Torch CPU/CUDA before unchanged native CLI modules.',
 'source_native_seed_or_repeat_flags_exist':False,'identical_native_or_published_seed_claim':False,
 'published_score_equality_claim':False,'current_policy_executed':False,'actual_optimizer_calls':0}
(O/'SEED_POLICY_REVISION_RECEIPT.json').write_text(json.dumps(seed_history,indent=2,sort_keys=True)+'\n')
remote=r'''
import base64,collections,gzip,hashlib,json,pathlib,subprocess,zlib
assert subprocess.check_output(['hostname'],text=True).strip()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
r=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');p=r/'training_pipeline_v2_20261012';packet=r/'ogb_batchnorm_author_cli_packet_20261012_v1'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def binding(path):return {'path':str(path.relative_to(r)),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())}
local_source=LOCAL_SOURCE_PAYLOAD
for row in local_source:
 path=p/row['path'];assert path.stat().st_size==row['bytes'] and sha(path.read_bytes())==row['sha256']
study_path=next((packet/'studies').glob('*.json.gz'));spec=json.loads(gzip.decompress(study_path.read_bytes()));plan=json.loads((packet/'PLAN.json').read_text())
pins={'main_pyg.py':(6894,'26125c957250accc8b933ab274ba6fa5ec5e21f91cd61df785266852fd1a156f'),'gnn.py':(2580,'eeac03fd138e49f0799a6757fc06f024f97d6774f814cba4ddb7ae5ff333f8b9'),'conv.py':(8741,'a592513e7fb0f6ed308ab036a961fe99382d3b09bd1fe3a503d5d8e65ff2e936'),'README.md':(2653,'0c4efb13e7a3166025f743a977e87f1af9cefaa9c814f7d1aad8ad0ef3fc0b7a')}
checked=[]
for run in spec['runs']:
 for row in run['cli']['source_files']:
  path=r/row['path'];assert binding(path)==row
  if path.name in pins:assert (row['bytes'],row['sha256'])==pins[path.name];checked.append(row)
assert len(checked)==320
assert len(spec['runs'])==len(plan['groups'])==80
assert not plan['scientific_launch_enabled'] and plan['human_GO_required']
assert not (packet/'execution').exists() and not (packet/'registry.sqlite').exists() and not (packet/'gate_receipts').exists()
assert not list((packet/'quarantine').iterdir())
assert all(not list((r/run['cli']['fresh_processed_dir']).iterdir()) for run in spec['runs'])
revision=SEED_HISTORY_PAYLOAD
assert spec['source']['replication_seed_policy']['seed_ids']==revision['accepted_fresh_seed_ids']
revision_path=packet/'SEED_POLICY_REVISION_RECEIPT.json';assert not revision_path.exists();revision_path.write_text(json.dumps(revision,indent=2,sort_keys=True)+'\n')
proof={'schema':'ogb-native-source-unchanged-and-disabled-metadata-proof-v1','author_commit':spec['source']['git_commit'],
 'native_source_pin_files':pins,'all80_process_native_source_files_rehashed':len(checked),'seed_wrapper_is_external':True,
 'native_source_CLI_modified':False,'source_code_path_replication_under_declared_seed_and_runtime':True,
 'native_or_published_seed_and_score_equality_claim':False,
 'recipe_seed_roster_counts':dict(collections.Counter(run['config']['dataset']+'-'+run['config']['source_model'] for run in spec['runs'])),
 'train_graph_authority':spec['data']['train_graph_count_authority'],
 'optimizer_call_upper_bound_derivation':{'MolHIV':{'TRAIN_graphs':32901,'batch_size':32,'ceil_batches_per_epoch':1029,'epochs':100,'processes':40,'maximum_optimizer_calls':4116000},
 'MolBACE':{'TRAIN_graphs':1210,'batch_size':32,'ceil_batches_per_epoch':38,'epochs':100,'processes':40,'maximum_optimizer_calls':152000},'total_maximum_optimizer_calls':4268000,'executed_optimizer_calls':0,
 'actual_future_calls_policy':'Native singleton-batch skips and failures can reduce calls. Public epoch observer does not claim exact optimizer calls.'},
 'TEST_quarantine_proof':{'observer':'run_ogb_author_cli.py:OGBProgress','captures':'Train and Validation ROC-AUC only','Test_float_conversion_or_dictionary_decoding':False,
 'raw_stdout_stderr_quarantine':'private per-process exclusive-create files','public_PROGRESS_RESULT_registry':'TRAIN/VALID/process/cost metadata only','release':'whole_author closure after exact whole-master HPO gate',
 'dispatch':'LIVE run_author_cli.execute dispatches dedicated OGB before PolyFormer preflight and Progress; PolyFormer parser untouched','focused_invalid_TEST_sentinel_fixture_passed':True},
 'study_receipt':binding(study_path),'plan_receipt':binding(packet/'PLAN.json'),
 'source_files':[binding(p/row['path']) for row in local_source],
 'seed_revision_receipt':binding(revision_path),'source_qualification':'Source/metadata/help/import only; actual native model/provider processing/resource/architecture qualification remains pending',
 'quarantine_files':0,'fresh_empty_processed_dirs':80,'registry_created':False,'gate_receipts_created':False,
 'model_constructions':0,'model_forwards':0,'scientific_fits':0,'optimizer_steps':0,'GO_created':False,'frozen_controller_changed':False,'root_master_changed':False}
proof_path=packet/'NATIVE_SOURCE_AND_DISABLED_METADATA_PROOF.json';assert not proof_path.exists();proof_path.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
payload={'proof':proof,'proof_receipt':binding(proof_path),'seed_revision_receipt':binding(revision_path)}
print(base64.b64encode(zlib.compress(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),9)).decode())
'''
remote=remote.replace('LOCAL_SOURCE_PAYLOAD',repr(source)).replace('SEED_HISTORY_PAYLOAD',repr(seed_history))
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(remote)]
run=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
(O/'SEAL_TRANSPORT.json').write_text(json.dumps({'exit_code':run.returncode,'stderr':run.stderr,'stdout_bytes':len(run.stdout)},indent=2)+'\n')
assert run.returncode==0,run.stdout[-3000:]
payload=json.loads(zlib.decompress(base64.b64decode(run.stdout.strip())))
(O/'NATIVE_SOURCE_AND_DISABLED_METADATA_PROOF.json').write_text(json.dumps(payload['proof'],indent=2,sort_keys=True)+'\n')
bindings=json.loads((O/'ALLOCATION_BINDINGS.json').read_text());bindings.update({k:payload[k] for k in ['proof_receipt','seed_revision_receipt']})
(O/'ALLOCATION_BINDINGS.json').write_text(json.dumps(bindings,indent=2,sort_keys=True)+'\n')
handoff={'schema':'ogb-author-family-source-packet-handoff-v1','owned_local_source_files':source,'allocation_packet':'ogb_batchnorm_author_cli_packet_20261012_v1',
 'allocation_plan_canonical_hash':'29816f48fad269aae62fcc02bd63669a687e10eeed0629eb656ae17fb1aef23b',
 'allocation_study_canonical_hash':'ac899ae71bfc8d7407122750a03d35697c7a17d94e317435c303288fc9410733',
 'allocation_bindings':bindings,'focused_local_tests':receipt(O/'FOCUSED_CPU_ENGINEERING_CHECK.json'),
 'native_source_proof':receipt(O/'NATIVE_SOURCE_AND_DISABLED_METADATA_PROOF.json'),
 'seed_policy_revision':receipt(O/'SEED_POLICY_REVISION_RECEIPT.json'),
 'finite_roster':'8 exact source recipes times10 independent declared seeds =80 external author subprocesses;0 executed',
 'maximum_optimizer_calls':4268000,'executed_optimizer_calls':0,
 'standalone_worker_entry':'training_pipeline_v2_20261012/run_ogb_author_cli.py',
 'live_dispatch_hooks_only':['author_cli_spec.py:validate_author_spec/bind_author_files/author_counts','run_author_cli.py:execute dedicated family branch'],
 'frozen_controller_or_master_or_shared_root_docs_or_publish_changed':False,
 'root_next_actions':['Audit GO/claim/private cwd/source/provider/quarantine/VALID selection and all proof bindings.',
 'Freeze a fresh successor runtime including all four new modules, existing author dispatch modules, registry/spec/run_study; update controller freeze manifest before master compile.',
 'Bind native provider-processing provenance/official split graph counts and complete exposure authority.',
 'Qualify actual BatchNorm GIN/GIN-virtual/GCN/GCN-virtual model state/resource/protocol and compatible original-GNNM/M1/genuine-I4 adaptation scope; current LayerNorm GINE does not establish these architecture gates.',
 'Replace provisional scheduling placeholders with root-qualified resource receipts in a fresh spec and integrate whole_search/whole_author gates into a fresh master.',
 'Human explicit GO remains required before any native fitting.']}
(O/'HANDOFF.json').write_text(json.dumps(handoff,indent=2,sort_keys=True)+'\n')
lines=['# OGB author-family implementation handoff','',
 'Dedicated validator, worker, external declared-seed entry, compiler and synthetic fixtures are implemented. Thin live author dispatch hooks are included; native source CLI and every frozen controller/master are unchanged. No shared root docs or publishing were changed during implementation.','',
 '**Finite disabled packet:** `ogb_batchnorm_author_cli_packet_20261012_v1` on allocation. Eight source recipes × ten independent seed processes =80 author groups. PLAN canonical `29816f48fad269aae62fcc02bd63669a687e10eeed0629eb656ae17fb1aef23b`; study canonical `ac899ae71bfc8d7407122750a03d35697c7a17d94e317435c303288fc9410733`. All source/raw bytes bound;80 fresh processed directories; empty quarantine; no registry or gate receipts.','',
 '## Owned local source files','']
for row in source:lines.append('- `'+str(P/row['path'])+'` — SHA256 `'+row['sha256']+'`')
lines+=['','The four new runtime modules must be included in the root successor controller freeze; live master/compiler freeze lists were not changed here.','',
 '## Source, seed and selection semantics','',
 'Pin `snap-stanford/ogb@61e9784ca76edeaa6e259ba0f836099608ff0586`. The320 native source files across80 process workdirs were rehashed against original main_pyg/gnn/conv/README pins. Native CLI bytes are unchanged. Exact32/300/5/Adam0.001/dropout0.5/100epoch/full-feature/scaffold recipes remain fixed.','',
 'The prior701xx root discussion remains a prospective historical policy. Root accepts73101,73103,73107,73109,73121,73127,73133,73141,73147,73151 as this fresh packet’s controlled prospective seed revision. Neither policy was executed. Native CLI has no seed/repeat flags: this is an external declared replication seed/runtime policy and source code-path replication, with no identical native/published seeds or published-score equality claim. See SEED_POLICY_REVISION_RECEIPT.json.','',
 'Every process has its own cwd/source/raw-only provider/quarantine and empty processed cache. Whole official VALID is selected by first np.argmax after all100 epochs. Public selected_epoch is the one-based native printed epoch; corresponding source curve index is selected_epoch−1. No model checkpoint/engine restore is invented. TEST-bearing original stdout/stderr remain quarantined until exact whole-author terminal closure, gated by whole-master search. The regex captures only Train/Validation tokens; no TEST float conversion or dictionary decoding occurs.','',
 '## Cost bounds and executed work','',
 '| Dataset | TRAIN | ceil(TRAIN/32) |100epochs ×40processes | Maximum optimizer calls |','|---|---:|---:|---:|---:|',
 '| MolHIV |32901 |1029 |4000epochs |4116000 |',
 '| MolBACE |1210 |38 |4000epochs |152000 |','',
 '**Maximum:4,268,000 calls. Executed:0.** Native singleton-batch skips and failures can reduce future actual calls; epoch stdout alone is not exact optimizer-call accounting. The observer leaves actual calls unresolved after a started process.','',
 'Focused checks:8 standard-library synthetic metadata/parser/custody tests passed,0 skipped. On verified allocation, dedicated worker/compiler/wrapper help, standard-library-only imports, all80 source/raw bindings and one process metadata preflight passed. No OGB GNN/provider constructor, forward, optimizer update, scientific fit, CUDA model work or GO occurred.','',
 '## Standalone disabled entry','',
 'Worker: `training_pipeline_v2_20261012/run_ogb_author_cli.py`. Study: `ogb_batchnorm_author_cli_packet_20261012_v1/studies/ogb-batchnorm-declared-seed-author-20261012-v1.json.gz`. First run: `ogbg-molhiv-gin-seed73101`. Use `--preflight-only` for metadata admission; fitting additionally requires root-qualified resource evidence, exact master/whole-search gate and explicit user GO. Do not launch any fit from this handoff.','',
 '## Remaining root gates','']
for action in handoff['root_next_actions']:lines.append('- '+action)
lines+=['', 'All detailed byte receipts and proof bindings are in HANDOFF.json, ALLOCATION_BINDINGS.json, NATIVE_SOURCE_AND_DISABLED_METADATA_PROOF.json and FOCUSED_CPU_ENGINEERING_CHECK.json. Earlier local pre-SSH command-length failure is retained as excluded; it performed no allocation access.','']
(O/'HANDOFF.md').write_text('\n'.join(lines))
files=['HANDOFF.json','HANDOFF.md','ALLOCATION_BINDINGS.json','NATIVE_SOURCE_AND_DISABLED_METADATA_PROOF.json','SEED_POLICY_REVISION_RECEIPT.json','FOCUSED_CPU_ENGINEERING_CHECK.json']
seal={'schema':'ogb-author-family-local-handoff-seal-v1','files':[receipt(O/name) for name in files],
      'source_files':source,'fitting_or_GO_enabled':False,'actual_optimizer_calls':0}
(O/'SEAL.json').write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n')
print(json.dumps({'handoff_path':str(O/'HANDOFF.md'),'handoff_SHA256':sha if False else hashlib.sha256((O/'HANDOFF.md').read_bytes()).hexdigest(),
 'seal_SHA256':hashlib.sha256((O/'SEAL.json').read_bytes()).hexdigest(),'native_source_proof_allocation_receipt':payload['proof_receipt']}))
