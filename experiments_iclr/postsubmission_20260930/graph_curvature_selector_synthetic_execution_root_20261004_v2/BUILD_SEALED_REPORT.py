"""Stdlib evidence packaging only: no numerical imports, data loads or launch."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def descriptor(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def seal(folder, scope):
    rows = []
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            row = descriptor(path)
            row['path'] = str(path.relative_to(folder))
            rows.append(row)
    write_json(folder/'MANIFEST.json', dict(schema='curvature-report-payload-manifest-v1',
        UTC=UTC, scope=scope, payload=rows))
    write_json(folder/'SEAL.json', dict(schema='curvature-report-seal-v1', UTC=UTC,
        artifact=folder.name, scope=scope, manifest='MANIFEST.json',
        manifest_sha256=descriptor(folder/'MANIFEST.json')['sha256'], payload_count=len(rows),
        scientific_or_predictive_qualification=False, permissions=dict(files='0444', directory='0555')))
    for path in folder.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(folder.rglob('*'), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    folder.chmod(0o555)
    return descriptor(folder/'SEAL.json')


cumulative = HERE/'cumulative_report'
cumulative.mkdir()
for name in ('FINAL_REPORT.md', 'FINAL_REPORT.json'):
    shutil.copyfile(HERE/name, cumulative/name)
receipt_names = ('SYNTHETIC_RESULT.json', 'EXECUTION_INPUT.json', 'EXECUTION_START.json',
    'EXECUTION_FINISH.json', 'PID_RECEIPT.json', 'SUITE_PID_RECEIPT.json',
    'STAGING_RECEIPT.json', 'GPU_UUID_CHECK.json', 'REMOTE_COLLECTION_01.json',
    'SUITE_STDOUT.txt', 'WRAPPER_STDOUT.txt', 'START_GUARD.json')
for version in ('v1', 'v2'):
    source = PHASE/f'graph_curvature_selector_synthetic_execution_root_20261004_{version}'
    target = cumulative/version
    target.mkdir()
    for name in receipt_names:
        shutil.copyfile(source/name, target/name)
    if version == 'v1':
        for name in ('EXECUTION_ANALYSIS.md', 'EXECUTION_ANALYSIS.json'):
            shutil.copyfile(source/name, target/name)

v1 = json.loads((cumulative/'v1/SUITE_PID_RECEIPT.json').read_text())
v2 = json.loads((cumulative/'v2/SUITE_PID_RECEIPT.json').read_text())
write_json(cumulative/'EXACT_EXECUTION_PATH.json', dict(schema='curvature-exact-execution-path-v1',
    route='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222',
    server_repository=REPO, server_phase=REMOTE_PHASE, local_phase=str(PHASE),
    sole_authorized_gpu_uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac',
    supervisor=dict(v1=400083, v2=400448), suite=dict(v1=v1, v2=v2),
    numerical_executions_per_version=dict(v1=1, v2=1), no_repeat_launch=True,
    local_evidence=dict(v1=str(PHASE/'graph_curvature_selector_synthetic_execution_root_20261004_v1'),
                        v2=str(HERE)),
    remote_evidence=dict(v1=REMOTE_PHASE+'/graph_curvature_selector_synthetic_execution_root_20261004_v1',
                        v2=REMOTE_PHASE+'/graph_curvature_selector_synthetic_execution_root_20261004_v2'),
    report_originals_preserved=True))
(cumulative/'EXACT_EXECUTION_PATH.md').write_text(f'''# Exact execution and evidence path

The authorized server route was `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222`, verified against sole UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` before remote file operations. Repository: `{REPO}`. Phase: `{REMOTE_PHASE}`.

One v1 suite and one v2 suite ran. V1 supervisor/suite PIDs: 400083/400085. V2: 400448/400450. The numerical interpreter was `{REPO}/.venv/bin/python`, with `-B`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1` and a project-local `tmp/` directory. Both suites used CPU tensors and reported CUDA unavailable.

The exact corrected-suite executable was `{REMOTE_PHASE}/graph_curvature_selector_synthetic_execution_root_20261004_v2/phase_tree/graph_curvature_selector_numerical_harness_preparation_20261004_v2/launch.py --report AUTHORIZED_SYNTHETIC_RESULT.json`. Its raw report was copied without numerical rerun to top-level `SYNTHETIC_RESULT.json`. Exact argv and environment are preserved in `v2/SUITE_PID_RECEIPT.json`; staged file hashes are in `v2/EXECUTION_INPUT.json`.

The guarded path was `LAUNCH_ONCE.py` → exclusive `START_GUARD.json` → detached `EXECUTE_ONCE.py` supervisor → one suite. No unknown SSH outcome triggered a repeat launch. Executed source packets remain in each server execution folder's `phase_tree/`; source/harness/constants hashes were checked before and after.

The matching original evidence roots are `{PHASE}/graph_curvature_selector_synthetic_execution_root_20261004_v1` and `{HERE}` locally, and the same folder names under the server phase. This `cumulative_report/` contains a separate sealed copy of cumulative reports, both raw results, failures/traces and custody receipts. It does not change either executed packet. All remaining actual-native or predictive work is source-only and unexecuted.
''')
cumulative_seal = seal(cumulative, 'Cumulative two-attempt synthetic CPU engineering evidence; native/predictive qualification unclaimed')

outline = PHASE/'graph_curvature_selector_native_followup_source_outline_20261004_v1'
outline.mkdir()
binding_file = PHASE/'modern_teacher_execution_root_v1/source_pack_audit_run03/SOURCE_LABEL_BINDING.json'
bindings = json.loads(binding_file.read_text())
graphs = []
for name in ('Squirrel', 'Photo'):
    metadata = PHASE/f'coordinate_conformal_execution_root_v1/acquisition_run02/{name}/GRAPH_INPUT.json'
    manifest = json.loads(metadata.read_text())
    cells = [dict(seed=row['seed'], source_split_index=(17,29,43).index(row['seed']),
        role_freeze=row['role_freeze'], source_labels=row['source_labels'])
        for row in bindings['pairs'] if row['graph_input']['path'].split('/')[-2] == name]
    graphs.append(dict(graph=name, graph_manifest=descriptor(metadata),
        server_graph_manifest=REMOTE_PHASE+f'/coordinate_conformal_execution_root_v1/acquisition_run02/{name}/GRAPH_INPUT.json',
        dimensions={key:manifest[key] for key in ('num_nodes','num_features','num_classes','num_edges')},
        features=manifest['features'], canonical_edges=manifest['edges'], cells=cells,
        metadata_declarations_only=True, remote_payload_availability_reverified=False,
        raw_arrays_or_labels_opened=False))
write_json(outline/'DECLARED_INPUT_PATHS.json', dict(schema='curvature-source-only-declared-input-paths-v1',
    UTC=UTC, source_label_metadata=descriptor(binding_file), graphs=graphs,
    restriction='Descriptors copied from local acquisition/label metadata only; no arrays, fitted states or outcomes opened. Existing scientific eligibility/release scope is not inherited.'))
write_json(outline/'SOURCE_PATHS.json', dict(schema='curvature-native-outline-source-paths-v1', UTC=UTC,
    phase=str(PHASE), server_phase=REMOTE_PHASE,
    sources=[descriptor(PHASE/path) for path in (
      'graph_curvature_selector_source_preparation_20261004_v3/driver.py',
      'graph_curvature_selector_source_preparation_20261004_v3/selector.py',
      'graph_curvature_selector_source_preparation_20261004_v3/SOURCE_BINDINGS.json',
      'graph_curvature_selector_prospective_constants_root_20261004_v1/FROZEN_CONSTANTS_AND_SCREEN.json',
      'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/modern_teacher_adapter.py',
      'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/backbone_boundary_adapter.py',
      'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/native_polyformer.py',
      'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/native_polyformer_outer.py',
      'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/native_polyformer_preprocess.py',
      'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/native_polynormer.py',
      'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision/prototype/graph_init_training_adapter.py',
      'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision/prototype/graph_band_route_initializer.py')],
    native_execution_launched=False, constants_or_selector_changed=False))
(outline/'NATIVE_QUALIFICATION_AND_30_ARM_OUTLINE.md').write_text('''# Actual-native qualification and fresh 30-arm development: source-only outline

Status: proposed path only. The completed 12-case CPU suite qualifies synthetic implementation engineering. No native body, fresh warm state, native memory/cost or predictive performance has been qualified. This outline launches nothing and changes no selector, candidate enumeration, constants, source warm/continuation schedule or checkpoint rule.

## Shortest scientifically faithful route

1. Use the sole authorized anogena-2 route and verify UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` before remote file operations. Bind the actual interpreter/runtime and versioned NumPy/PyTorch/SciPy/PyG plus required sparse/scatter operators; native preprocessing and models import dependencies absent from the synthetic coverage. The v3 loader must return the identical verified native adapter used for acquisition, including executed hash/path tags. No old fitted state or source certificate establishes the new actual-state result.
2. Read only the declared full raw FP32 feature/canonical int64 edge arrays and compact TRAIN/VALIDATION packs within their existing scope. Preserve `derived_roles_v2`, graph identity and seed17/29/43 ↔ split0/1/2. Use all nodes and edges. Recompute preprocessing with the bound adapter; no persistent cache or graph/name-only cache reuse.
3. First acquire fresh representative cfg0 `single_author` seed17 states on Squirrel and Photo using the unchanged `warm_native` function. Squirrel needs all 50 fixed-last updates. Photo needs all 200 local updates, earliest strict validation-NLL-best local model/Adam restored, RNG retained after all 200, then all 50 global updates. A shorter warm run or an old donor is a different state and cannot qualify the prescribed endpoint.
4. On disposable copies of each actual endpoint, qualify serialized model/spec/stage/alias/Adam moments/steps/options and exact RNG restoration, full-graph native/K1/each-K4 identity logits, and native/K4 one-step correspondence using `optimizer_equivalence_audit`. That correspondence audit deliberately freezes new R/S. Separately require actual selector/trial and one-step continuation checks with every shared/private R/S live, finite gradients and correctly normalized own-member TRAIN loss. Preserve inactive Photo parameters and their moment custody too.
5. Bind only the final predictive head (`head.R`, width256 on Squirrel; `global_head.R`, width512 on Photo), with K1 in eval mode. Run the existing `qualify_gradient_interface` at this actual endpoint: three fixed directions, JVP/VJP duality and centered-logit/CE finite differences at epsilons1e-3 and3e-4, existing tolerances. Then run `prepare_from_native_warm` once for that block: one common trial and all nine candidate slots across original graph, permuted graph and random span, at most ten actual coupled-Adam maps. Preserve rank/D/cap/finite/trial failures and all nulls or abstentions; never redraw, change radius/cap, move warm point or relax tolerances to obtain a non-null result.
6. Check all five returned models, named optimizers and RNG states against independently reconstructed pre-trial initializations. Trials are discarded; each arm starts from the saved original continuation RNG and correct pre-trial Adam history. Check exact restoration of modes/aliases/state and no trial residue, including real selected/fixed non-common arms if naturally eligible. If none is eligible, report missing native non-common coverage; do not invent it. Measure wall time, process CPU time, host peak RSS and, only under a later CUDA execution scope, synchronized allocated/reserved GPU peak for cold preprocessing, warm, qualification, construction/matching, every attempted trial, and continuation. Record failures and charge every attempt.
7. After actual-state correctness and practical full-schedule resource feasibility are established, use the two newly acquired unchanged seed17 warm checkpoints as seed17 development starts; rebuild preparation from their exact saved RNG and retain qualification cost. Acquire the other four warm starts freshly for seed29/43, checking per-block identity/state/gradient custody at each start. Continue each block's five returned arms under the unchanged native schedule: six blocks × five arms =30 continuations. Retain all attempted blocks, null arms, failures and costs. No shortened predictive pilot or outcome-dependent replacement block.

These steps qualify engineering and execution feasibility first. Any native failure must be retained and resolved with a separately preserved source correction before predictive continuation. This outline adds no registry or approval framework.

## Representative native recipes

| Item | Squirrel | Photo |
|---|---|---|
| Native body | PolyFormer-Mono | Polynormer-r |
| Full graph | N2223, F2089, C5, 46998 canonical undirected edges | N7650, F745, C8, 119081 canonical undirected edges |
| Architecture | hidden256; monomial order12 /13 tokens; layers2; heads4; d_ffn128; q1.4; multi1 | heads8 × hidden64 =512; local7/global2 layers; beta−1; pre_ln false; qk_shared true |
| Native dropout | dropout.3, dprate.8 | input.2, local.7, global.7 |
| Coupled Adam | base lr1e-4, decay0; attention lr1e-3, decay1e-7 | lr.001, decay5e-5 |
| Fresh native warm | 50 fixed-last native updates | all200 local, best-local model/Adam + post-all200 RNG, then50 fixed-last global |
| Per-arm continuation | cap1950; native patience250 | 950 global updates |
| Frozen midpoint | continuation950 / native epoch1000, if early stopping permits | continuation450 / global epoch500 / actual update700 |

Squirrel preprocessing is raw features, native `to_undirected`, GCN normalization and SciPy COO FP32 sparse arithmetic, materializing X through Ahat^12X as `[2223,13,2089]`. That token tensor alone is 241480524 bytes (about230.3MiB), before transient copies, activations, gradients, optimizer states or five returned K4 models. Photo applies PyG `NormalizeFeatures` exactly once to the bound provider feature values, makes edges undirected, and removes/adds self loops once. Canonical source edges remain distinct from teacher edges.

## Served accuracy and NLL

The training objective remains the arithmetic mean of the four own-member TRAIN cross-entropies, with native noisy training. The served probabilities are `softmax(mean raw member logits)`; served accuracy is the argmax of those mean logits. Report served accuracy alongside pooled NLL and Brier, with member accuracy/NLL for competence. Checkpoint selection stays pooled VALIDATION NLL, earliest strict tie, with continuation epoch0 eligible.

Report all six paired blocks and each arm's selected endpoint and prescribed terminal/fixed-final endpoint. The current continuation function emits selected and frozen-midpoint logits, then restores the NLL-best model; it does not emit terminal logits before restoration. A future observation-only reporting wrapper must retain terminal logits/metrics at the last completed update without changing training, RNG, stopping or selection. Mark an absent midpoint when Squirrel's native patience stops before it; do not fabricate a fixed cap endpoint that was never reached. Record actual updates and costs.

Keep the frozen exploratory mechanism screen verbatim: graph-wise mean selected-validation NLL improvement at least.01 nats versus every common/fixed/permuted/random control, favorable paired NLL signs in all three seeds on each graph, and graph-wise mean accuracy decline no more than.005. This is selection-associated validation evidence; the mechanism gate does not establish accuracy superiority. Predictive accuracy through ensembles remains the broader primary objective, and storage/cold/warm serving cost are secondary reports. No new split or held-out set is introduced. Do not use A/B/D/correction labels or open final-pool labels without their already declared release scope.

## Exact source and declared data locations

Local phase: `/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930`.
Server phase: `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930`.

`SOURCE_PATHS.json` supplies exact local source paths/hashes. Under either phase, use `graph_curvature_selector_source_preparation_20261004_v3/driver.py` and `selector.py`; unchanged native adapter/body/boundary/preprocessing under `continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3/prototype/`; state/warm/continuation and graph construction under `continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision/prototype/graph_init_training_adapter.py` and `graph_band_route_initializer.py`.

`DECLARED_INPUT_PATHS.json` supplies graph and all six role/TRAIN/VALIDATION descriptors with declared hashes. Server graph root: `<server phase>/coordinate_conformal_execution_root_v1/acquisition_run02/{Squirrel,Photo}/`; inputs `GRAPH_INPUT.json`, `features.npy`, `edges.npy`; roles `prepared_roles/seed{17,29,43}_split{0,1,2}/ROLE_FREEZE.json`, `train_nodes.npy`, `validation_nodes.npy`; compact label packs `labels/source/seed{seed}_split{index}_{train,validation}.npz`. Only the local JSON metadata was read for this outline. Payload existence/content has not been reverified here. Those input descriptors are data provenance, not permission to reuse an old warm checkpoint, result, certificate or closed-study allocation.

Frozen constants file: `graph_curvature_selector_prospective_constants_root_20261004_v1/FROZEN_CONSTANTS_AND_SCREEN.json`, SHA256 `633ea814bfc82c094ecb2d98699e77258f0f644d13cf17fa7f8d65691cf637dc`. Radius.5/cap1; rank atol1e-10/rtol1e-6; sign atol1e-12; mean-logit atol1e-5/rtol1e-6; D floor1e-9, match atol1e-10/rtol.001; bisection40; tie/abstention1e-6. Source v3 manifest `d53d6963b3c66cb59061c4f5be6a81108adf8e9792a3f3cf93daea56fe35ec70`.

## CPU/GPU suitability

CPU supports the tested primitive/state logic. Actual native CPU coverage still needs installed/version-bound SciPy/PyG/operators and AD qualification. The full native warm/30-arm schedule is much heavier than the 4.88s toy suite; no empirical CPU time/RSS forecast is available. Whole-graph native CUDA execution is the practical candidate if resources and scope permit, with actual-device AD and native custody checked on that device. CPU toy correctness does not qualify CUDA kernels, determinism, memory or capacity.

Only the sole authorized UUID is known; its GPU model, memory, free capacity and availability were not measured by this work. The root's co-resident graph-view CUDA QA currently owns the GPU. This outline does not query or interfere with that job, reserve GPU resources, signal processes or launch any native work. A future plan must use its real measured full-graph warm/trial/continuation costs; no GPU feasibility claim follows from UUID verification.
''')
outline_seal = seal(outline, 'Source-only future actual-native qualification and 30-arm outline; no numerical/native execution')

# Package only the cumulative report for the already authorized report save.
records=[]
for path in sorted(cumulative.rglob('*')):
    if path.is_file():
        data=path.read_bytes()
        records.append(dict(path=str(path.relative_to(HERE)), bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), base64=base64.b64encode(data).decode()))
remote = 'RECORDS = '+repr(records)+'''\nimport base64,hashlib,json,pathlib,subprocess
uuids=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert [u.strip() for u in uuids if u.strip()]==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=pathlib.Path('''+repr(REMOTE_PHASE+'/'+HERE.name)+''')
for record in RECORDS:
    path=root/record['path']
    data=base64.b64decode(record['base64'])
    assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    if path.exists(): assert path.read_bytes()==data, 'Refuse differing report: '+str(path)
for record in RECORDS:
    path=root/record['path']
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(base64.b64decode(record['base64']))
    assert path.stat().st_size==record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    path.chmod(0o444)
folder=root/'cumulative_report'
for path in sorted(folder.rglob('*'),reverse=True):
    if path.is_dir(): path.chmod(0o555)
folder.chmod(0o555)
manifest=folder/'MANIFEST.json'
seal=folder/'SEAL.json'
print(json.dumps(dict(status='CUMULATIVE_REPORT_SEALED',GPU_UUID=uuids[0].strip(),files=len(RECORDS),
    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),
    no_differing_files_overwritten=True,remote_report=str(folder))))
'''
(HERE/'SAVE_CUMULATIVE_REMOTE.py').write_text(remote)
write_json(HERE/'LOCAL_REPORT_SEAL_RECEIPT.json', dict(UTC=UTC,
    cumulative_report=str(cumulative), cumulative_seal=cumulative_seal,
    native_outline=str(outline), native_outline_seal=outline_seal,
    no_native_or_numerical_execution=True))
print(json.dumps(dict(cumulative_report=str(cumulative),cumulative_seal=cumulative_seal,
    native_outline=str(outline),native_outline_seal=outline_seal,remote_report_files=len(records))))
