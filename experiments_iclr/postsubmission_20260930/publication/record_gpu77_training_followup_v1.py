"""Publish locally fetched v5 qualification metadata; never run experiments.

Root invocation: python3 publication/record_gpu77_training_followup_v1.py
    --snapshot coordination_snapshots/<fresh-name>
The prospective root README is written inside that fresh snapshot, not at root.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


PHASE = Path(__file__).resolve().parents[1]
REPO77 = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
REMOTE_PHASE = REPO77 + '/experiments_iclr/postsubmission_20260930'
PYTHON77 = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python'
UUIDS = ['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
         'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
ARMS = ['native1024', 'single256', 'factorized4', 'independent4', 'matched_single']
SOURCE = 'buddy_shared_cache_execution_v5'
DATA = 'buddy_complete_data_cache_preparation_v3'
LAUNCHER = 'buddy_gpu77_resource_family_launcher_v2'
PACKET_SHAS = {
    SOURCE: 'c479cedbd244ff645c7ee822625120b9f809d5082f4d9d4ace265237b5712e0f',
    DATA: '8469488ee3338b4488b0ff310fe23840a91080dc6b5939ff350d6092abeda348',
    LAUNCHER: 'af494ae88dd264f512475d60b4e571d20fd9285ebe81e32217127dcc0aef0323',
}
CPU = 'gpu77_buddy_numerical_qualification_v2/root_run_v1/CPU_QUALIFICATION.json'
CPU_RUN = 'gpu77_buddy_numerical_qualification_v2/root_run_v1/CPU_RUN_RECEIPT.json'
DATA_RUN = DATA + '/root_run_77_v1/QUALIFICATION.json'
CACHE = DATA + '/root_run_77_v1/cache/manifest.json'
RESOURCE = LAUNCHER + '/root_resource_v1/RESOURCE_RECEIPT.json'
ADMISSION = LAUNCHER + '/ROOT_ADMISSION.json'
FAMILY = LAUNCHER + '/root_family_v1/FAMILY_LAUNCH_RECEIPT.json'
FAMILY_START = LAUNCHER + '/root_family_launch_v1/FAMILY_START.json'
README_BASE = 'coordination_snapshots/20261002_gpu77_buddy_v4_followup_v2/README.md'
FAILURE = 'buddy_gpu77_resource_family_launcher_v1/root_resource_v1/native1024.log'
PROBE = 'gpu77_connection_recovery_v1/commands/cuda77_initialization_check_v1/RECEIPT.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    return json.loads((PHASE / relative).read_text())


def evidence(relative):
    return {'path': relative, 'sha256': sha(PHASE / relative)}


def matches(actual, expected, label):
    require(all(actual.get(key) == value for key, value in expected.items()),
            label + ' identity/status differs')


def verify_packet(relative):
    folder = PHASE / relative
    manifest = read(relative + '/SOURCE_MANIFEST.json')
    digest = sha(folder / 'SOURCE_MANIFEST.json')
    require(digest == PACKET_SHAS[relative], relative + ' manifest differs')
    matches(read(relative + '/SEAL.json'), {'source_manifest_sha256': digest}, relative + ' seal')
    for item in manifest['files']:
        path = folder / item['path']
        require(path.resolve().is_relative_to(folder.resolve()), 'Source path escapes packet')
        require(sha(path) == item['sha256'] and path.stat().st_size == item['bytes'],
                relative + '/' + item['path'] + ' byte differs')
    return {item['path']: item['sha256'] for item in manifest['files']}


def qualification():
    sources = verify_packet(SOURCE)
    verify_packet(DATA)
    verify_packet(LAUNCHER)
    implementation = {name: digest for name, digest in sources.items()
                      if (name.endswith('.py') and '/' not in name) or name.startswith('vendor/')
                      or name in {'CONFIG.json', 'SOURCE_PINS.json', 'requirements-qualification-extra.txt'}}
    cpu, run, data, cache = read(CPU), read(CPU_RUN), read(DATA_RUN), read(CACHE)
    matches(cpu, dict(status='synthetic_cpu_pass', test_count=7, dataset_access=False,
                      gpu_execution=False, implementation_hashes=implementation,
                      test_script_sha256=sources['test_cpu.py']), 'CPU certificate')
    require(cpu['torch_version'].split('+')[0] == '2.7.1', 'Actual77 Torch version differs')
    matches(run, dict(status='SEVEN_CPU_CHECKS_PASSED', exit_code=0, certificate=cpu,
                      actual_git_root=REPO77, source_manifest_sha256=PACKET_SHAS[SOURCE],
                      GPU_compute=False, dataset_access=False, other_jobs_stopped=False), 'CPU execution receipt')
    require(len(run['physical_GPU_UUIDs']) == 2 and set(run['physical_GPU_UUIDs']) == set(UUIDS),
            'Actual77 physical GPU inventory differs')
    matches(data, dict(status='complete_official_train_valid_cache_qualified',
                       builder_manifest_sha256=PACKET_SHAS[SOURCE],
                       wrapper_manifest_sha256=PACKET_SHAS[DATA], CPU_qualification_sha256=sha(PHASE / CPU),
                       Torch_version=cpu['torch_version'], cpu_threads=4, GPU_compute=False,
                       OGB_legacy_data_load_environment='TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
                       test_members_opened=False, combined_split_accessor_called=False,
                       cache_output=REMOTE_PHASE + '/' + DATA + '/root_run_77_v1/cache',
                       cache_manifest_sha256=sha(PHASE / CACHE), qualification_before_hashing=True),
            'Complete-data qualification')
    matches(data['public_graph'], dict(node_count=235868, raw_edge_rows=1179052,
                                       directed_entries=2358104), 'Complete public graph')
    matches(data['train_topology'], dict(status='complete_train_topology_pass', train_positive_rows=1179052,
                                         train_max_year=2017, coalesced_directed_entries=1935264,
                                         raw_train_record_multiset_equal=True), 'Training topology')
    matches(data['validation'], dict(positive_rows=60084, negative_rows=100000), 'Validation metadata')
    required_contents = ['all_nodes_retained', 'all_structural_rows_finite', 'native_zero_columns_4_5_exact',
                         'native_hash_bank_complete_finite', 'degrees_equal_full_weighted_train_oracle',
                         'candidate_positive_and_validation_negative_order_exact',
                         'training_negatives_unique_train_disjoint_and_nonself',
                         'future_positives_never_opened_or_filtered']
    require(all(data['cache_contents'].get(key) is True for key in required_contents), 'Cache qualification failed')
    matches(cache, dict(dataset='ogbl-collab', graph_policy='training_only_all_splits', node_count=235868,
                        coalesced_directed_entries=1935264, implementation_hashes=implementation,
                        test_split_opened=False, config_sha256=sources['CONFIG.json'],
                        source_pin_sha256=sources['SOURCE_PINS.json'], builder_sha256=sources['cache_builder.py']),
            'Cache manifest')
    require(set(cache['files']) == {'common.pt', 'train.pt', 'valid.pt', 'official_split_loader.py.txt'},
            'Cache manifest membership differs')
    # This producer validates fetched metadata only; it never opens tensor/cache payloads.
    for split in ['train', 'valid']:
        require(cache['official_split_files'][split]['sha256'] == data['qualified_official_split_files'][split],
                'Qualified official split binding differs')
    matches(cache, dict(edge_index_sha256=data['train_topology']['coalesced_edge_index_sha256'],
                        edge_weight_sha256=data['train_topology']['coalesced_edge_weight_sha256'],
                        raw_features_sha256=data['public_graph']['raw_features_tensor_sha256']), 'Cache topology binding')
    config = read(SOURCE + '/CONFIG.json')
    matches(config, dict(arms=ARMS, seeds=[0, 1, 2], epochs=100, graph_policy='training_only_all_splits'), 'Fixed family')
    repair = read(SOURCE + '/REPAIR_PROOF.json')
    require(sha(PHASE / FAILURE) == repair['failure_log_sha256'] ==
            'fada5b2ec5eb918f12ac747acdfbe5fc5d9f7ba2f9494571614ad744d6b0404e', 'Preserved v4 failure differs')
    require(sha(PHASE / PROBE) == repair['root_data_free_cuda_probe_sha256'] ==
            'f0d7eb8f02e1715780cac482934c7bcfb33fe9ee5244a95b9ed590471d7ee3f2', 'CUDA repair probe differs')
    return cpu, run, data


def launcher_evidence(cpu):
    expected_context = dict(wrapper_manifest_sha256=PACKET_SHAS[LAUNCHER],
                            source_manifest_sha256=PACKET_SHAS[SOURCE],
                            CPU_qualification_sha256=sha(PHASE / CPU),
                            CPU_execution_receipt_sha256=sha(PHASE / CPU_RUN),
                            data_qualification_sha256=sha(PHASE / DATA_RUN),
                            cache_manifest_sha256=sha(PHASE / CACHE), actual_git_root=REPO77,
                            torch_version=cpu['torch_version'], cpu_threads=4,
                            physical_GPU_UUIDs=UUIDS, implementation_hashes=cpu['implementation_hashes'])
    result = dict(resource_status='not_locally_available', resource_complete=False,
                  admission_locally_available=False, family_start_locally_available=False,
                  family_receipt_locally_available=False, current_running_state_verified=False)
    resource = None
    if (PHASE / RESOURCE).exists():
        resource = read(RESOURCE)
        matches(resource, dict(schema='buddy77-all-arm-resource-receipt-v1', other_jobs_stopped=False,
                                test_access=False), 'Resource receipt')
        matches(resource['context'], expected_context, 'Resource context')
        require(resource['status'] in {'in_progress', 'failed', 'all_five_complete_resource_epochs'},
                'Unrecognized resource status')
        result.update(resource_status=resource['status'], resource_receipt=evidence(RESOURCE))
        if resource['status'] == 'all_five_complete_resource_epochs':
            matches(resource, dict(resource_optimizer_fits=8, family_cells=15, family_optimizer_fits=24,
                                    family_epochs_per_cell=100, family_validation_forwards=1500,
                                    prospective_root_admission_required=True), 'Complete resource schedule')
            require([row['arm'] for row in resource['commands']] == ARMS and
                    all(row['exit_code'] == 0 and row['GPU_UUID'] == UUIDS[i % 2]
                        for i, row in enumerate(resource['commands'])), 'All five successful resource commands required')
            require([row['arm'] for row in resource['resources']] == ARMS, 'All five resource evidence rows required')
            rows = []
            for row in resource['resources']:
                clean = {'arm': row['arm']}
                for field, filename in [('completion_sha256', 'completion.json'),
                                        ('identity_sha256', 'identity.json'), ('epochs_sha256', 'epochs.jsonl')]:
                    require(re.fullmatch('[0-9a-f]{64}', row[field]) is not None, 'Resource evidence digest missing')
                    clean[field] = row[field]
                    local = PHASE / LAUNCHER / 'root_resource_v1' / row['arm'] / filename
                    if local.exists():
                        require(sha(local) == row[field], 'Locally fetched resource metadata byte differs')
                # Do not inspect or publish the receipt's training-loss/outcome record.
                rows.append(clean)
            result.update(resource_complete=True, resources=rows)
    if (PHASE / ADMISSION).exists():
        require(result['resource_complete'], 'Local admission requires complete matching local resource evidence')
        admission = read(ADMISSION)
        matches(admission, dict(schema='buddy77-prospective-family-admission-v1', decision='admitted',
                                 source_manifest_sha256=PACKET_SHAS[SOURCE], wrapper_manifest_sha256=PACKET_SHAS[LAUNCHER],
                                 cache_manifest_sha256=sha(PHASE / CACHE), CPU_qualification_sha256=sha(PHASE / CPU),
                                 resource_receipt_sha256=sha(PHASE / RESOURCE), family_cells=15,
                                 optimizer_fits=24, cpu_threads=4, physical_GPU_UUIDs=UUIDS), 'Prospective root admission')
        require(bool(admission.get('root_resource_cost_decision')), 'Root resource-cost decision missing')
        result.update(admission_locally_available=True, admission=evidence(ADMISSION))
    if (PHASE / FAMILY_START).exists():
        require(result['admission_locally_available'], 'Local detached START requires matching prospective admission')
        start = read(FAMILY_START)
        expected_start_argv = [PYTHON77, '-B', REMOTE_PHASE + '/' + LAUNCHER + '/launch77.py',
                               'family', '--execute', '--admission', REMOTE_PHASE + '/' + ADMISSION]
        matches(start, dict(schema='buddy77-detached-family-start-v1', status='detached_start_only',
                             argv=expected_start_argv, actual_git_root=REPO77,
                             source_manifest_sha256=PACKET_SHAS[SOURCE], wrapper_manifest_sha256=PACKET_SHAS[LAUNCHER],
                             admission_sha256=sha(PHASE / ADMISSION), resource_receipt_sha256=sha(PHASE / RESOURCE),
                             physical_GPU_UUIDs=UUIDS, family_cells=15, optimizer_fits=24, cpu_threads=4,
                             other_jobs_stopped=False, test_access=False, terminal_receipt_relative_path=FAMILY),
                'Detached family START receipt')
        require(type(start['PID']) is int and start['PID'] > 0, 'Detached START PID missing')
        result.update(family_start_locally_available=True, family_start_receipt=evidence(FAMILY_START),
                      detached_family_start_PID=start['PID'])
        # A historical launch PID attests dispatch, never current liveness or completion.
    if (PHASE / FAMILY).exists():
        require(result['admission_locally_available'], 'Local family receipt requires matching prospective admission')
        family = read(FAMILY)
        matches(family['context'], expected_context, 'Family context')
        expected_argv = [PYTHON77, '-B', REMOTE_PHASE + '/' + SOURCE + '/launch_family.py',
                         '--cache', REMOTE_PHASE + '/' + DATA + '/root_run_77_v1/cache',
                         '--runs', REMOTE_PHASE + '/' + LAUNCHER + '/root_family_v1/runs',
                         '--qualification', REMOTE_PHASE + '/' + CPU, '--gpus', ','.join(UUIDS), '--execute']
        matches(family, dict(schema='buddy77-family-launch-receipt-v1', admission_sha256=sha(PHASE / ADMISSION),
                              resource_receipt_sha256=sha(PHASE / RESOURCE), argv=expected_argv,
                              family_cells=15, optimizer_fits=24, other_jobs_stopped=False, test_access=False),
                'Family launch receipt')
        require(type(family['exit_code']) is int, 'Family exit code missing')
        result.update(family_receipt_locally_available=True, family_receipt=evidence(FAMILY),
                      family_launcher_exit_code=family['exit_code'])
    return result


def literature():
    version = 'index_v19' if (PHASE / 'literature_memory/index_v19/LITERATURE_INDEX.json').is_file() else 'index_v18'
    relative = 'literature_memory/' + version + '/LITERATURE_INDEX.json'
    index = read(relative)
    records = index['paper_records']
    groups = index['canonical_identifier_normalization']['groups']
    kinds = Counter(group['kind'] for group in groups)
    require(set(kinds) <= {'paper', 'software_documentation'}, 'Literature group kind differs')
    require(len({group['normalized_identifier'] for group in groups}) == len(groups), 'Duplicate literature groups')
    coverage = []
    for group in groups:
        for position in group['record_indices']:
            require(type(position) is int and 0 <= position < len(records), 'Literature record index differs')
            require(records[position]['canonical_id'] in group['raw_canonical_identifiers'], 'Literature group binding differs')
            coverage.append(position)
    require(sorted(coverage) == list(range(len(records))), 'Literature groups must cover every record exactly once')
    counts = dict(conclusion_records=len(records), normalized_paper_identifiers=kinds['paper'],
                  software_documentation_identifiers=kinds['software_documentation'])
    matches(index['read_accounting'], counts, 'Literature metadata counts')
    require(index['read_accounting']['full_paper_read_total_certified'] is False,
            'This follow-up must not certify a full-paper-read total')
    return dict(path='literature_memory/' + version, index_sha256=sha(PHASE / relative),
                **counts, full_paper_read_total_certified=False)


def resource_prose(launch):
    if launch['family_receipt_locally_available']:
        return ('A locally fetched receipt binds the fixed family invocation to the complete resource epochs and '
                f'prospective root admission; the family launcher returned exit code {launch["family_launcher_exit_code"]}. '
                'Individual fitted results and scientific comparisons have not been audited by this follow-up.')
    if launch['family_start_locally_available']:
        return ('A locally fetched detached START receipt attests the authorized family launcher launch after all five '
                'complete resource epochs and separate prospective root admission. No terminal family receipt is locally '
                'available yet. The launch PID does not establish current liveness, fitted completion or training progress; '
                'current progress requires separate evidence. Scientific comparisons remain unaudited.')
    if launch['admission_locally_available']:
        return ('All five complete resource epochs and a matching separate prospective root admission are locally attested. '
                'No local detached START or terminal family receipt is available yet.')
    if launch['resource_complete']:
        return ('All five complete resource epochs are locally attested. Separate prospective root admission is still required '
                'before the fixed family; no matching local admission or family receipt is available.')
    if launch['resource_status'] == 'failed':
        return ('The v5 resource wrapper has a locally fetched failure receipt. All-five resource completion and family '
                'admission are not established; this failure is retained with the earlier v4 failure.')
    return ('All-five resource completion is not yet attested by a locally fetched receipt. All five complete resource epochs '
            'and separate prospective root admission precede the fixed family.')


def replace_section(text, heading, body, end_heading=None):
    marker = '## ' + heading + '\n'
    require(text.count(marker) == 1, 'Expected one section: ' + heading)
    start = text.index(marker)
    if end_heading:
        end = text.index('## ' + end_heading + '\n', start + len(marker))
    else:
        next_heading = re.search(r'^## ', text[start + len(marker):], re.MULTILINE)
        end = start + len(marker) + next_heading.start() if next_heading else len(text)
    return text[:start] + marker + '\n' + body.strip() + '\n\n' + text[end:]


def status_text(before, now, run, launch, memory):
    checkout = ('Two NVIDIA A100 80 GB devices were verified. Existing jobs remain active and no other jobs were stopped. '
                'The existing parent source/archive files were preserved. The exact-v5 CPU receipt binds the checkout '
                f'`{REPO77}` at Git head `{run["git_head"]}` during qualification; later syncs have their own receipts.')
    text, changes = re.subn(r'^Two NVIDIA A100 80 GB devices were verified\.[^\n]*', checkout,
                           before, count=1, flags=re.MULTILINE)
    require(changes == 1, 'Checkout status paragraph missing')
    runtime = ('The saved runtime is Python 3.12.11, Torch 2.7.1 and PyG 2.4.0. '
               'BUDDY dependencies are installed in an isolated project-local target. '
               'All seven exact-v5 synthetic CPU checks passed on actual 18.77; this qualification used no GPU or dataset. '
               'The user permits our jobs alongside existing GPU activity; no other jobs were stopped. '
               'Shared-host contention prevents isolated timing or speedup claims.')
    text, changes = re.subn(r'^The saved runtime is[^\n]*', runtime, text, count=1, flags=re.MULTILINE)
    require(changes == 1, 'Runtime status paragraph missing')
    link = (f'BUDDY source v5 repairs CUDA device initialization immediately before peak-memory reset. '
            'The v4 resource failure (Torch 2.7 Invalid device argument before model construction, optimizer or training) '
            f'is preserved at `{FAILURE}`. Config, vendor recipe, CPU tests and cache builder are unchanged. '
            f'All seven exact-v5 CPU checks passed on actual 18.77 with Torch 2.7.1 in {run["seconds"]:.2f} seconds.\n\n'
            'The complete official ogbl-collab training/validation cache qualification passed for all 235,868 nodes. '
            'Topology and cached graph features use training data only; official validation pairs are retained in order. '
            'No test member or combined split accessor was opened. All arms reuse this one deterministic cache.\n\n'
            + resource_prose(launch) + '\n\n'
            'The registered schedule remains five fixed arms × three seeds × 100 epochs (15 cells, 24 optimizer fits). '
            'The process-local OGB compatibility environment is retained; explicit checkpoint loading still uses weights_only=True. '
            'Other jobs remain active, so observed resource timing and memory depend on contention. '
            'This setup establishes no isolated speedup, new predictive result or methodological novelty.')
    text = replace_section(text, 'Link prediction', link)
    literature_paragraph = (f'Literature {Path(memory["path"]).name} contains {memory["conclusion_records"]} conclusion records '
                            f'across {memory["normalized_paper_identifiers"]} normalized paper identifiers and '
                            f'{memory["software_documentation_identifiers"]} software documentation identifiers. '
                            'These counts were checked against the index metadata and normalization groups. '
                            'Scoped method reads, retained notes and revisits are not full-paper reads; a cumulative full-paper-read '
                            'total is not certified. Prior conclusions and unresolved novelty limitations remain retained.')
    marker = '## Literature and scientific claims\n'
    start = text.index(marker) + len(marker)
    paragraph_start = start + len(text[start:]) - len(text[start:].lstrip('\n'))
    paragraph_end = text.index('\n\n', paragraph_start)
    text = text[:paragraph_start] + literature_paragraph + text[paragraph_end:]
    text, changes = re.subn(r'^Updated: [^\n]*', 'Updated: ' + now + '. The research goal remains incomplete.',
                           text, count=1, flags=re.MULTILINE)
    require(changes == 1, 'Status timestamp missing')
    return text


def prospective_readme(base, launch, memory):
    body = ('The original five-dataset benchmark and paper scores remain unchanged. The fixed graph-initialization study '
            'remains active on modern backbones, comparing graph-filtered training-error initialization with common descent, '
            'random tangent directions, altered topology and warm copying. Canonical coordinator receipts own its progress; '
            'final labels remain closed until the registered cohort and admissions are complete. No new predictive advantage '
            'or methodological novelty is established.\n\n'
            'The additional 18.77 host has two A100 80 GB GPUs and an exact Git checkout. Project-local dependencies are installed. '
            'The user permits concurrent jobs; other jobs remain running. The seven-GPU account serves only as the explicitly '
            'authorized MacLink forwarding relay. Shared-host contention prevents isolated speed or memory-efficiency claims.\n\n'
            'BUDDY source v5 repairs CUDA initialization before peak-memory reset; the v4 resource failure is preserved. '
            'All seven exact-v5 CPU checks passed on actual 18.77 with Torch 2.7.1. Full official data/cache qualification passed '
            'for 235,868 nodes using training-only topology, with validation pairs preserved and test members unopened. '
            'The five fixed arms share one deterministic cache and retain three seeds × 100 epochs (15 cells, 24 optimizer fits). '
            + resource_prose(launch) + '\n\n'
            'All 14 restricted industrial native imports passed after the documented read-only OS entropy repair. '
            'Those imports and the stronger node-classification control fitter are setup evidence; full-context model/data/GPU '
            'qualification, fitted comparisons and efficiency measurement remain pending. RelBench temporal sources are prepared. '
            'Unsuccessful studies and engineering failures remain preserved; stored parameters alone do not establish serving efficiency.\n\n'
            'Read [current status](experiments_iclr/postsubmission_20260930/PUBLIC_STATUS.md), '
            f'[literature memory](experiments_iclr/postsubmission_20260930/{memory["path"]}/README.md), and '
            '[review obligations](experiments_iclr/postsubmission_20260930/REVIEW_RESPONSE_TRACKER.md). '
            f'Literature memory contains {memory["conclusion_records"]} conclusion records across '
            f'{memory["normalized_paper_identifiers"]} normalized paper identifiers and '
            f'{memory["software_documentation_identifiers"]} software documentation identifiers. '
            'These are checked index counts, not full-paper-read totals. Scientific claims require complete evidence and fresh '
            'independent paper reviews. No new result, revised manuscript or acceptance claim is made.')
    return replace_section(base, 'Post-submission research', body, 'Repository contents')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, help='Fresh name or path under coordination_snapshots/')
    args = parser.parse_args()
    supplied = Path(args.snapshot)
    snapshots = PHASE / 'coordination_snapshots'
    out = supplied if supplied.is_absolute() else (snapshots / supplied if len(supplied.parts) == 1 else PHASE / supplied)
    out = out.absolute()
    require(out == out.resolve() and out.parent == snapshots.resolve() and not out.exists(),
            '--snapshot must name a fresh direct directory under coordination_snapshots/')
    cpu, run, data = qualification()
    launch = launcher_evidence(cpu)
    memory = literature()
    now = datetime.now(timezone.utc).isoformat()
    before_ledger_text = (PHASE / 'research_ledger.json').read_text()
    ledger = json.loads(before_ledger_text)
    before_status = {name: (PHASE / name).read_text() for name in ['PUBLIC_STATUS.md', 'RESEARCH_STATE.md']}
    after_status = {name: status_text(text, now, run, launch, memory) for name, text in before_status.items()}
    base = (PHASE / README_BASE).read_text()
    readme = prospective_readme(base, launch, memory)
    if launch['family_receipt_locally_available']:
        state = 'buddy_v5_gpu77_family_launcher_returned_' + str(launch['family_launcher_exit_code']) + '_results_unaudited'
    elif launch['family_start_locally_available']:
        state = 'buddy_v5_gpu77_family_detached_start_attested_terminal_not_locally_available'
    elif launch['admission_locally_available']:
        state = 'buddy_v5_gpu77_resources_complete_prospective_family_admitted'
    elif launch['resource_complete']:
        state = 'buddy_v5_gpu77_resources_complete_family_admission_pending'
    elif launch['resource_status'] == 'failed':
        state = 'buddy_v5_gpu77_CPU_and_full_cache_pass_resource_failure_preserved'
    else:
        state = 'buddy_v5_gpu77_CPU_and_full_cache_pass_resource_completion_not_locally_attested'
    record = dict(schema='gpu77-buddy-training-publication-followup-v1', UTC=now, status=state,
                  snapshot=out.relative_to(PHASE).as_posix(), source_manifest_sha256=PACKET_SHAS[SOURCE],
                  data_wrapper_manifest_sha256=PACKET_SHAS[DATA], launcher_manifest_sha256=PACKET_SHAS[LAUNCHER],
                  CPU_certificate=evidence(CPU), CPU_execution_receipt=evidence(CPU_RUN),
                  data_qualification=evidence(DATA_RUN), cache_manifest=evidence(CACHE),
                  passed_actual77_CPU_tests=7, torch_version=cpu['torch_version'],
                  full_node_count=data['public_graph']['node_count'], training_only_topology=True,
                  test_members_opened=False, other_jobs_stopped=False, isolated_speedup_established=False,
                  v4_resource_failure=evidence(FAILURE), data_free_CUDA_repair_probe=evidence(PROBE),
                  fixed_family=dict(arms=ARMS, seeds=[0, 1, 2], epochs=100, cells=15, optimizer_fits=24),
                  launcher_evidence=launch, literature_memory=memory, outcome_metrics_inspected=False,
                  scientific_results_audited=False, original_scores_changed=False,
                  predictive_advantage_claimed=False, methodological_novelty_claimed=False,
                  prospective_root_README=str(out / 'README.md'), README_base=evidence(README_BASE),
                  README_replacement_scope='Only Post-submission research; Repository contents and all other bytes preserved',
                  audit_and_automation_records_preserved=True)
    ledger['current_status_update_UTC'] = now
    ledger['status'] = state
    ledger['latest_literature_memory'] = dict(memory)
    ledger['literature_memory_current'] = dict(memory)
    ledger.setdefault('gpu77_buddy_training_followups_v1', []).append(record)
    ledger['gpu77_buddy_training_followup_v1'] = record
    ledger.setdefault('postsubmission_event_log', []).append(dict(
        UTC=now, event='gpu77_buddy_v5_training_followup', snapshot=record['snapshot'],
        status=state, final_labels_read=False, predictive_advantage_claimed=False))
    after_ledger = json.dumps(ledger, indent=2, allow_nan=False) + '\n'
    # Validate everything before creating the fresh packet or changing current state.
    out.mkdir(exist_ok=False)
    (out / 'before').mkdir()
    (out / 'before/research_ledger.json').write_text(before_ledger_text)
    for name, text in before_status.items():
        (out / 'before' / name).write_text(text)
    (out / 'README.md').write_text(readme)
    (out / 'FOLLOWUP.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    (out / 'research_ledger.json').write_text(after_ledger)
    for name, text in after_status.items():
        (out / name).write_text(text)
    (PHASE / 'research_ledger.json').write_text(after_ledger)
    for name, text in after_status.items():
        (PHASE / name).write_text(text)
    print(json.dumps(dict(status=state, snapshot=str(out), prospective_root_README=str(out / 'README.md'),
                          evidence_paths=[CPU, CPU_RUN, DATA_RUN, CACHE, FAILURE, PROBE] +
                          [path for path in [RESOURCE, ADMISSION, FAMILY_START, FAMILY] if (PHASE / path).exists()],
                          CPU_tests=7, cache_nodes=235868, resource_status=launch['resource_status'],
                          prospective_admission=launch['admission_locally_available'],
                          detached_family_start=launch['family_start_locally_available'],
                          family_receipt=launch['family_receipt_locally_available'], literature=memory['path']), sort_keys=True))


if __name__ == '__main__':
    main()
