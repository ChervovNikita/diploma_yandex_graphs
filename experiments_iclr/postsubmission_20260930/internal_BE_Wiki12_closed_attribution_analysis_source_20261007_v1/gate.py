"""Saved all12/owner/lane custody gate. Stdlib only; no live polling or scoring."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101, 6203, 6307)
CONDITIONS = ('plain', 'alignment_only', 'residual_only', 'combined')
MAX_FORWARDS = 48


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Phase-relative custody required')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Path leaves authorized phase')
    return path


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and path.is_file(), 'Phase file custody')
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Bound file changed: ' + row['path'])
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Bound file size changed')
    return path


def verify(row):
    path = bound(row)
    for item in read(path)['files']:
        file = path.parent / item['path']
        require(file.is_file() and sha(file) == item['sha256'] and file.stat().st_size == item['bytes'], 'Sealed source changed')
    return path.parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def consume(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact separate root release')
    cfg = read(release_path); pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(cfg.get('schema') == 'Wiki12-closed-attribution-analysis-release-v1', 'Exact release schema')
    for key in ('enabled', 'root_execution_authorized', 'source_review_approved', 'entire12_complete',
                'owner_lanes_and_children_terminal', 'trusted_checkpoint_deserialization_authorized',
                'runtime_resource_readiness_confirmed', 'collection_and_analysis_cost_charged', 'external_owned_bound_confirmed'):
        require(cfg.get(key) is True, 'Disabled pending root release: ' + key)
    require(cfg.get('TEST_access') is False and cfg.get('reselection') is False and cfg.get('automatic_retry') is False
            and cfg.get('maximum_member_forwards') == MAX_FORWARDS, 'Fixed no-TEST/no-reselection/no-retry collection')
    require(cfg['analysis_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Reviewed analysis source changed')
    verify(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['analysis_manifest_sha256']))
    for row in pins['references']:
        bound(row)
    verify(pins['training_source_manifest']); verify(pins['public_manifest']); verify(pins['audited_Wiki24_analysis_manifest'])
    source_pins = read(bound(pins['training_source_bindings']))
    bundle = read(bound(pins['scientific_bundle']))
    require(bundle['enabled'] and bundle['fixed12_protocol_adopted']
            and bundle['source_manifest_sha256'] == pins['training_source_manifest']['sha256']
            and bundle['owner_sha256'] == pins['scientific_owner']['sha256'], 'Exact frozen scientific bundle/source/owner')
    family_path = bound(cfg['family_closure']); family = read(family_path)
    activation = inside(pins['activation_directory'])
    require(family_path == activation / 'FAMILY_CLOSURE.json' and family.get('complete') is True
            and family.get('lane_results') == [True, True] and family.get('fixed_scientific_cells') == 12
            and family.get('TEST_access') is False and family.get('automatic_retry') is False, 'Whole12 complete original family closure')
    require(not any((activation / ('LANE_' + str(index) + '_FAILURE.json')).exists() for index in (0, 1)), 'No failed lane admitted')
    parent_path = bound(cfg['parent_owner']); parent = read(parent_path)
    require(parent_path == activation / 'PARENT_OWNER.json' and isinstance(parent, dict)
            and type(parent.get('PID')) is int and type(parent.get('start_ticks')) is int
            and str(bound(pins['scientific_owner'])) in parent.get('argv', []), 'Original saved parent owner identity')
    terminal = read(bound(cfg['terminal_evidence']))
    require(terminal.get('schema') == 'Wiki12-saved-terminal-evidence-v1'
            and terminal.get('owner_lanes_and_children_terminal') is True
            and terminal.get('family_closure') == cfg['family_closure'] and terminal.get('parent_owner') == cfg['parent_owner']
            and terminal.get('parent_identity') == parent and terminal.get('parent_absent') is True
            and terminal.get('parent_no_CUDA_rows') is True, 'Root terminal custody for this exact parent/closure')
    children = terminal['children']; require(len(children) == 12, 'Entire12 terminal children evidence')
    child_by_cell = {row['cell_id']: row for row in children}; require(len(child_by_cell) == 12, 'Unique terminal child roster')
    records = []
    expected_counts = dict(shadow_member_forwards=8800, replay_member_forwards=8800,
        output_cotangent_collections=1100, member_reverse_collections=8800, optimizer_bank_updates=1100,
        exact_member_RNG_endpoint_checks=1100)
    for index in (0, 1):
        lane_path = bound(cfg['lane_closures'][str(index)]); lane = read(lane_path)
        items = bundle['lanes'][str(index)]
        require(lane_path == activation / ('LANE_' + str(index) + '_COMPLETE.json') and lane.get('complete') is True
                and [row['cell_id'] for row in lane['completed']] == [row['cell_id'] for row in items], 'Exact whole completed lane')
        for item, closed in zip(items, lane['completed']):
            cell_cfg = read(bound(item)); condition, seed = cell_cfg['condition'], cell_cfg['seed']
            cell = str(seed) + '_' + condition
            require(condition in CONDITIONS and seed in SEEDS and item['cell_id'] == cell and closed['cell_id'] == cell
                    and cell_cfg['source_manifest_sha256'] == pins['training_source_manifest']['sha256']
                    and cell_cfg['train'] == source_pins['train'] and cell_cfg['development'] == source_pins['development']
                    and cell_cfg['physical_gpu_uuid'] == source_pins['GPU_per_seed'][str(seed)], 'Fixed cell source/data/GPU identity')
            receipt = closed['exit_receipt']; identity = receipt['raw_identity_observation']
            expected_argv = [source_pins['python'], '-B', str(bound(pins['training_adapter'])), '--release', str(bound(item)), '--release-sha256', item['sha256']]
            require(identity is not None and identity['argv'] == expected_argv and identity['pgid'] == identity['sid'] == identity['PID'], 'Exact original isolated scientific child argv/session')
            exit_path = activation / 'logs' / (cell + '.EXIT.json')
            require(read(exit_path) == receipt and receipt['cell_id'] == cell and receipt['job_sha256'] == item['sha256']
                    and receipt['exit_code'] == 0 and receipt['terminal_wait_observed'] is True
                    and receipt['reason'] is None and receipt['signals_sent'] == [] and receipt['attempts'] == 1
                    and receipt['retry'] is False and closed['complete'] is True
                    and closed['child_absent'] is True and closed['child_no_CUDA_rows'] is True, 'Original clean one-attempt owned exit')
            child = child_by_cell[cell]
            require(child['identity'] == identity and child['child_absent'] is True and child['child_no_CUDA_rows'] is True
                    and child['terminal_wait_observed'] is True, 'Root exact terminal child identity')
            output = inside(cell_cfg['output']); complete_path = output / 'COMPLETE.json'; run_path = output / 'RUN.json'
            require(not (output / 'FAILURE.json').exists() and sha(complete_path) == closed['completion_sha256'], 'Original complete endpoint custody')
            endpoint, run = read(complete_path), read(run_path)
            label = 'be_unit__mechanism_v2_' + condition
            require(endpoint['complete'] is True and endpoint['epochs'] == endpoint['steps'] == 1100
                    and endpoint['seed'] == seed and endpoint['task'] == 'wikics' and endpoint['arm'] == label
                    and endpoint['method_identity'] == run['method_identity'] == label
                    and endpoint['mechanism_ablation']['condition'] == run['mechanism_ablation']['condition'] == condition
                    and endpoint['execution_accounting'] == expected_counts
                    and endpoint['ablation_adapter_sha256'] == run['ablation_adapter_sha256'] == pins['training_adapter']['sha256']
                    and endpoint['TEST_scoring'] is False and run['TEST_scoring'] is False, 'Exact complete condition execution identity')
            require(run['data']['train_npz_sha256'] == source_pins['train']['sha256']
                    and run['data']['valid_npz_sha256'] == source_pins['development']['sha256'], 'Original run role hashes')
            selected = output / 'selected.pt'
            require(sha(selected) == endpoint['selected_sha256'], 'Original selected snapshot hash; no reselection')
            records.append(dict(cell=cell, seed=seed, condition=condition, method_identity=label,
                release=binding(bound(item)), complete=binding(complete_path), run=binding(run_path),
                exit_receipt=binding(exit_path), selected_checkpoint=binding(selected), family_status='complete', members=4))
    roster = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
    require(len(records) == 12 and {(row['seed'], row['condition']) for row in records} == roster
            and set(child_by_cell) == {row['cell'] for row in records}, 'Entire fixed12 roster complete before numerical opening')
    bound(source_pins['train']); bound(source_pins['development'])
    bound(cfg['runtime_evidence']); bound(cfg['resource_readiness']); bound(cfg['external_supervision'])
    require(0 < cfg['external_active_seconds'] <= 3600 and cfg['external_cleanup_seconds'] == 10
            and cfg['external_hard_seconds'] == cfg['external_active_seconds'] + 10, 'Finite externally owned analysis envelope')
    require(0 < cfg['owned_GPU_cap_bytes'] <= 32*1024**3
            and cfg['minimum_fresh_GPU_free_bytes'] >= cfg['owned_GPU_cap_bytes'] + 2*1024**3, 'Root reviewed analysis GPU cap and fresh headroom')
    output = inside(cfg['output_directory'])
    require(not output.exists() and output.parent.is_dir(), 'Fresh server-only output; no retry/resume')
    return cfg, pins, source_pins, records, output
