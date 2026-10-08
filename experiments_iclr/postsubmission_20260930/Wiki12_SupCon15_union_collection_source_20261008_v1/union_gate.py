"""Stdlib metadata gate for immutable failed old owner + unused2 union + SupCon3."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101, 6203, 6307)
CONDITIONS = ('plain', 'alignment_only', 'residual_only', 'combined', 'supcon_eq2')
MAX_FORWARDS = 60
WORK = dict(shadow_member_forwards=8800, replay_member_forwards=8800,
    output_cotangent_collections=1100, member_reverse_collections=8800,
    optimizer_bank_updates=1100, exact_member_RNG_endpoint_checks=1100)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def consume(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact separate root collection release')
    cfg = read(release_path)
    require(cfg.get('schema') == 'Wiki12-SupCon15-union-collection-release-v1', 'Exact collection release schema')
    for key in ('enabled', 'root_execution_authorized', 'source_review_approved', 'entire_Wiki12_union_complete',
                'entire_SupCon3_complete', 'owners_and_all15_children_terminal', 'trusted_checkpoint_deserialization_authorized',
                'runtime_resource_readiness_confirmed', 'collection_and_analysis_cost_charged', 'external_owned_bound_confirmed'):
        require(cfg.get(key) is True, 'Disabled pending root admission: ' + key)
    require(all(cfg.get(key) is False for key in ('TEST_access', 'training', 'reselection', 'calibration', 'automatic_retry'))
            and cfg.get('maximum_member_forwards') == MAX_FORWARDS, 'Fixed prediction-only scope')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    # The reused helper module is stdlib only; its Wiki12 consume() is never called.
    row = pins['reuse']['legacy_gate']; path = PHASE / row['path']
    require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Original metadata helpers changed')
    spec = importlib.util.spec_from_file_location('gate', path)
    gate = importlib.util.module_from_spec(spec); sys.modules['gate'] = gate; spec.loader.exec_module(gate)
    gate.verify(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['collection_manifest_sha256']))
    for row in pins['reuse'].values():
        gate.bound(row)
    for key in ('legacy_manifest', 'original_manifest', 'public_manifest', 'supcon_manifest',
                'supcon_controller_manifest', 'unused2_manifest'):
        gate.verify(pins['reuse'][key])
    gate.verify(pins['audited_Wiki24_analysis_manifest']); gate.bound(pins['audited_Wiki24_analysis'])
    roster = pins['roster']; by_cell = {row['cell']: row for row in roster}
    require(len(roster) == len(by_cell) == 15 and {(r['seed'], r['condition']) for r in roster}
            == {(s, c) for s in SEEDS for c in CONDITIONS}, 'Exact fixed fifteen states')
    original = gate.inside(pins['original_activation']); fresh = gate.inside(pins['unused2_output'])
    supcon = gate.inside(pins['supcon_output'])
    old = read(gate.bound(pins['old_terminal_custody']))
    require(old['old_owner_absent'] is True and old['quality_scores_read'] is False
            and old['checkpoint_loaded'] is False, 'Frozen old10 metadata custody')
    for path, row in old['files'].items():
        gate.bound(dict(row, path=path))
    require(read(original / 'FAMILY_CLOSURE.json')['complete'] is False
            and read(original / 'LANE_0_FAILURE.json')['error']
            == 'TimeoutError: Fixed fresh-GPU resource window exhausted before child launch',
            'Preserve original failed closure/resource wait as immutable evidence')
    union_path = gate.bound(cfg['Wiki12_union_closure']); union = read(union_path)
    require(union_path == fresh / 'UNION_CLOSURE.json'
            and union['schema'] == 'Wiki12-old10-plus-never-started2-union-closure-v1'
            and union['complete'] is True and union['original_fixed_cells'] == 12
            and union['original_completed_cells'] == 10 and union['old_terminal_custody'] == pins['old_terminal_custody']
            and union['original_complete_false_closure_immutable'] is True
            and union['original_completed_cells_not_repeated'] is True and union['never_started_cells_only'] is True
            and union['automatic_retry'] is False and union['quality_scores_read'] is False
            and union['TEST_access'] is False and union['fatal_error'] is None,
            'Explicit successful old10 + unchanged never-started2 union required')
    failed = union['preserved_original_failed_preflight']; preflight = read(gate.bound(failed))
    require(gate.bound(failed) == original / 'logs/6307_residual_only.PREFLIGHT.json'
            and preflight == failed['observation'] and preflight['scientific_child_started'] is False
            and preflight['elapsed_seconds'] >= 1800, 'Retain original failed admission and charged wait')
    old_rows = {row['cell_id']: row for row in old['completed']}
    new_rows = {row['cell_id']: row for row in union['new_completed']}
    require(len(old_rows) == 10 and len(new_rows) == 2 and set(new_rows) == {'6307_residual_only', '6307_combined'}
            and set(old_rows) == {row['cell'] for row in roster if row['custody_partition'] == 'old10'}
            and [row['release']['cell_id'] for row in old['unused']] == ['6307_residual_only', '6307_combined']
            and all(row['scientific_child_started'] is False and row['output_absent'] is True for row in old['unused']),
            'Exact original ten plus two previously unused cells')
    for cell in new_rows:
        require(not (original / 'logs' / (cell + '.CHILD_STARTED.json')).exists(), 'No original trained-cell retry')
    family_path = gate.bound(cfg['SupCon3_closure']); family = read(family_path)
    family_binding = gate.binding(family_path)
    require(family_path == supcon / 'CLOSURE.json'
            and family['schema'] == 'canonical-SupCon-full3-owned-family-closure-v1'
            and family['family_accounted'] is True and family['all_new_fits_complete'] is True
            and family['fatal_error'] is None and family['family_hard_cap_exceeded'] is False
            and family['automatic_retry'] is False and family['quality_or_prediction_opening_performed'] is False
            and family['controller_manifest_sha256'] == pins['reuse']['supcon_controller_manifest']['sha256']
            and family['protocol_sha256'] == pins['reuse']['supcon_protocol']['sha256'], 'Exact original completed canonical3 family')
    require(union['supcon_terminal_custody']['path'] == cfg['SupCon3_closure']['path']
            and union['supcon_terminal_custody']['sha256'] == cfg['SupCon3_closure']['sha256']
            and union['supcon_terminal_custody']['all_new_fits_complete'] is True, 'Continuation binds this same original SupCon terminal family')
    terminal = read(gate.bound(cfg['terminal_evidence']))
    require(terminal.get('schema') == 'Wiki12-SupCon15-root-saved-terminal-evidence-v1'
            and terminal.get('complete') is True and terminal.get('root_observed') is True
            and terminal.get('owners_and_all15_children_terminal') is True
            and terminal['Wiki12_union_closure'] == cfg['Wiki12_union_closure']
            and terminal['SupCon3_closure'] == cfg['SupCon3_closure']
            and terminal['old_terminal_custody'] == pins['old_terminal_custody']
            and terminal['quality_scores_read'] is False and terminal['TEST_access'] is False,
            'Later root snapshot of the complete union and canonical family')
    owners = {row['role']: row for row in terminal['owners']}
    require(len(terminal['owners']) == len(owners) == 3 and set(owners) == set(pins['owners']), 'All three owner histories retained')
    owner_files = dict(original_Wiki12=original / 'PARENT_OWNER.json', unused2=fresh / 'OWNER.json', SupCon3=supcon / 'OWNER.json')
    parent_identities = {}
    for role, path in owner_files.items():
        saved = read(path); identity = saved if role == 'original_Wiki12' else saved['owner']
        pid_key = 'pid' if role == 'SupCon3' else 'PID'
        expected = pins['owners'][role]
        require(identity[pid_key] == expected['pid'] and identity['start_ticks'] == expected['start_ticks']
                and owners[role]['identity'] == identity and owners[role]['owner_receipt'] == gate.binding(path)
                and owners[role]['absent'] is True and owners[role]['no_CUDA_rows'] is True, 'Exact saved owner terminal custody: ' + role)
        parent_identities[role] = identity
        if role == 'unused2':
            require(saved['admission_sha256'] == pins['reuse']['unused2_admission']['sha256']
                    and saved['protocol'] == pins['reuse']['unused2_protocol'], 'Exact original unused2 source/admission owner')
        elif role == 'SupCon3':
            require(saved['admission_sha256'] == pins['reuse']['supcon_admission']['sha256']
                    and saved['controller_manifest_sha256'] == pins['reuse']['supcon_controller_manifest']['sha256'],
                    'Exact original canonical controller admission')
    require(family['owner'] == parent_identities['SupCon3'], 'Original canonical controller identity')
    children = {row['cell']: row for row in terminal['children']}
    require(len(terminal['children']) == len(children) == 15 and set(children) == set(by_cell), 'All15 exact terminal child slots')
    source_pins = pins['original_source_pins']; records = []
    core_root = gate.bound(pins['reuse']['public_manifest']).parent
    core = {name: sha(core_root / 'core' / name) for name in ('factors.py', 'models.py', 'objectives.py', 'selection.py')}
    supcon_rows = {row['seed']: row for row in family['rows']}
    require(len(family['rows']) == len(supcon_rows) == 3 and set(supcon_rows) == set(SEEDS), 'All fixed canonical seed slots')
    for item in roster:
        cell, seed, condition = item['cell'], item['seed'], item['condition']
        output = gate.inside(item['output']); complete_path, run_path = output / 'COMPLETE.json', output / 'RUN.json'
        require(not (output / 'FAILURE.json').exists(), 'No failed fit admitted: ' + cell)
        endpoint, run = read(complete_path), read(run_path)
        require(endpoint['complete'] is True and endpoint['epochs'] == endpoint['steps'] == 1100
                and endpoint['execution_accounting'] == WORK
                and all(row['task'] == 'wikics' and row['seed'] == seed and row['arm'] == item['method_identity']
                    and row['method_identity'] == item['method_identity'] and row['TEST_scoring'] is False for row in (endpoint, run))
                and run['core'] == core and run['native']['polynormer_model_sha256'] == source_pins['polynormer']['sha256']
                and run['data']['train_npz_sha256'] == source_pins['train']['sha256']
                and run['data']['valid_npz_sha256'] == source_pins['development']['sha256'], 'Original full1100 method/source/data: ' + cell)
        record = dict(cell=cell, seed=seed, condition=condition, method_identity=item['method_identity'], family_status='complete',
            collection_status='pending', members=4, complete=gate.binding(complete_path), run=gate.binding(run_path),
            custody_partition=item['custody_partition'], selected_path=output / 'selected.pt', selected_sha256=endpoint['selected_sha256'])
        if condition != 'supcon_eq2':
            release = gate.bound(item['release']); cell_cfg = read(release)
            require(cell_cfg['condition'] == condition and cell_cfg['seed'] == seed and cell_cfg['output'] == item['output']
                    and cell_cfg['source_manifest_sha256'] == pins['reuse']['original_manifest']['sha256']
                    and cell_cfg['train'] == source_pins['train'] and cell_cfg['development'] == source_pins['development']
                    and cell_cfg['physical_gpu_uuid'] == source_pins['GPU_per_seed'][str(seed)]
                    and endpoint['mechanism_ablation']['condition'] == run['mechanism_ablation']['condition'] == condition
                    and endpoint['ablation_adapter_sha256'] == run['ablation_adapter_sha256'] == pins['reuse']['original_adapter']['sha256'],
                    'Original Wiki12 release and semantic identity')
            closed = old_rows[cell] if item['custody_partition'] == 'old10' else new_rows[cell]
            require(closed['release'] == item['release'], 'Unchanged original release in union')
            exit_path = (gate.bound(closed['exit']) if item['custody_partition'] == 'old10'
                         else fresh / 'logs' / (cell + '.EXIT.json'))
            receipt = read(exit_path); identity = receipt['raw_identity_observation']
            expected_argv = [source_pins['python'], '-B', str(gate.bound(pins['reuse']['original_adapter'])),
                             '--release', str(release), '--release-sha256', item['release']['sha256']]
            require(identity['argv'] == expected_argv and identity['pgid'] == identity['sid'] == identity['PID']
                    and receipt['exit_code'] == 0 and receipt['reason'] is None and receipt['terminal_wait_observed'] is True
                    and receipt['signals_sent'] == [] and receipt['job_sha256'] == item['release']['sha256'], 'Exact original scientific wait/identity')
            if item['custody_partition'] == 'old10':
                require(sha(complete_path) == closed['completion']['sha256'] and closed['exit']['child_absent'] is True
                        and gate.bound(closed['completion']) == complete_path
                        and closed['exit']['pid'] == identity['PID']
                        and closed['exit']['child_no_CUDA_rows'] is True, 'Original old10 endpoint/exit binding')
            else:
                require(sha(complete_path) == closed['completion_sha256'] and sha(exit_path) == closed['exit_sha256']
                        and receipt == closed['exit_receipt'] and closed['complete'] is True
                        and closed['child_absent'] is True and closed['child_no_CUDA_rows'] is True, 'Explicit successful unchanged unused2 endpoint/exit')
            record.update(release=gate.binding(release), exit_receipt=gate.binding(exit_path))
        else:
            closed = supcon_rows[seed]; identity = closed['owner']; job_path = gate.inside(item['job']); job = read(job_path)
            require(closed['status'] == 'complete' and closed['exit_code'] == 0 and closed['actual_exit_and_reap'] is True
                    and closed['owned_child_after'] is None and closed['hard_cap_exceeded'] is False
                    and sha(complete_path) == closed['completion_sha256'] and endpoint['selected_sha256'] == closed['selected_sha256']
                    and sha(job_path) == closed['job_sha256'] and job['parent_owner'] == parent_identities['SupCon3']
                    and job['output'] == str(output) and job['seed'] == seed
                    and job['physical_gpu_uuid'] == source_pins['GPU_per_seed'][str(seed)]
                    and job['controller_manifest_sha256'] == pins['reuse']['supcon_controller_manifest']['sha256']
                    and job['protocol_sha256'] == pins['reuse']['supcon_protocol']['sha256']
                    and job['admission_sha256'] == pins['reuse']['supcon_admission']['sha256'], 'Exact canonical original worker/job custody')
            expected_argv = [source_pins['python'], '-B', str(gate.bound(pins['reuse']['supcon_entry'])),
                             '--job', str(job_path), '--job-sha256', closed['job_sha256']]
            require(closed['argv'] == expected_argv and identity['pid'] == identity['group'] == identity['session'], 'Original canonical worker session/argv')
            entry_path = supcon / 'entry' / ('seed' + str(seed) + '_TERMINAL.json'); entry = read(entry_path)
            started = read(supcon / 'entry' / ('seed' + str(seed) + '_STARTED.json'))
            require(sha(entry_path) == closed['entry_terminal_sha256'] and entry['exit_code'] == 0
                    and entry['job_sha256'] == closed['job_sha256'] and entry['error'] is None
                    and entry['controller_manifest_sha256'] == pins['reuse']['supcon_controller_manifest']['sha256']
                    and entry['protocol_sha256'] == pins['reuse']['supcon_protocol']['sha256']
                    and started['owner'] == identity and started['parent_owner'] == parent_identities['SupCon3']
                    and started['source_manifest_sha256'] == pins['reuse']['supcon_manifest']['sha256']
                    and started['physical_gpu_uuid'] == source_pins['GPU_per_seed'][str(seed)], 'Original canonical entry wait/source/device')
            definition = endpoint['public_loss_comparison']
            require(definition == run['public_loss_comparison'] and definition['condition'] == 'supcon_eq2'
                    and definition['public_wrapper_sha256'] == pins['reuse']['supcon_adapter']['sha256']
                    and definition['loss_module_sha256'] == pins['reuse']['supcon_loss']['sha256']
                    and definition['recompute_sha256'] == pins['reuse']['supcon_recompute']['sha256']
                    and definition['public_interface_manifest_sha256'] == pins['reuse']['public_manifest']['sha256']
                    and definition['loss_definition'] == 'canonical_SupCon_Eq2_outside_log_adaptation'
                    and endpoint['active_loss_calls'] == dict(alignment=1100, residual=0)
                    and endpoint['part_of_registered_author_Wiki12'] is False, 'Distinct canonical loss and complete execution')
            expected_versions = {name: pins['runtime'][name] for name in ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')}
            require(endpoint['actual_provider_versions'] == expected_versions
                    and run['data']['ordered_projected_arrays_match_public_pins'] is True, 'Matched canonical original runtime and ordered roles')
            record.update(job=gate.binding(job_path), entry_terminal=gate.binding(entry_path), parent_admission=pins['reuse']['supcon_admission'])
        child = children[cell]
        require(child['identity'] == identity and child['child_absent'] is True and child['child_no_CUDA_rows'] is True
                and child['wait_and_reap_observed'] is True
                and child['source_receipt'] == (record['exit_receipt'] if 'exit_receipt' in record else family_binding),
                'Root exact scientific child terminal custody: ' + cell)
        records.append(record)
    # All families/endpoints/owners/children are now closed. Hash streams do not deserialize tensors.
    for row in records:
        row['selected_checkpoint'] = gate.binding(row.pop('selected_path'))
        require(row['selected_checkpoint']['sha256'] == row.pop('selected_sha256'), 'Original selected snapshot; no reselection')
    for key in ('train', 'development', 'polynormer'):
        gate.bound(source_pins[key])
    for key in ('runtime_evidence', 'resource_readiness', 'external_supervision'):
        gate.bound(cfg[key])
    require(type(cfg['external_active_seconds']) is int and 0 < cfg['external_active_seconds'] <= 3600
            and cfg['external_cleanup_seconds'] == 10 and cfg['external_hard_seconds'] == cfg['external_active_seconds'] + 10
            and type(cfg['owned_GPU_cap_bytes']) is int and 0 < cfg['owned_GPU_cap_bytes'] <= 32 * 1024**3
            and cfg['minimum_fresh_GPU_free_bytes'] >= cfg['owned_GPU_cap_bytes'] + 2 * 1024**3,
            'Finite reviewed external/resource envelope; root enforces wall/cleanup bounds')
    output = gate.inside(cfg['output_directory'])
    roots = {Path(row['path']).parts[0] for row in pins['reuse'].values()}
    roots.update(Path(row['path']).parts[0] for row in (
        [pins['audited_Wiki24_analysis']] + [source_pins[key] for key in ('train', 'development', 'polynormer')]))
    roots.update((HERE.name, pins['original_activation'], pins['unused2_output'], pins['supcon_output'],
                  'wikics_unit_mechanism_gpu77_execution_root_20261007_v2'))
    require(not output.exists() and output.parent.is_dir()
            and not any(output.is_relative_to(gate.inside(root)) for root in roots), 'Fresh server-only output outside frozen sources/families')
    return cfg, pins, source_pins, records, output, gate
