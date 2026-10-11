"""Stdlib custody and finite native Penn94 protocol. No automatic admission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ARMS = ('internal_shared4', 'original_boundary_shared4', 'matched_boundary_shared4',
        'ordinary_M1', 'internal_factor_M1', 'genuine_internal_factor_I4', 'fixed_LINKX')
SEEDS = tuple(range(10))  # Proposal only until root admission; no outcome-driven top-up.
ALTERNATES = ('internal_shared4', 'matched_boundary_shared4')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def route():
    require(socket.gethostname() == 'anogena-2-0', 'Literal authorized allocation hostname required')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
            == [UUID], 'Sole authorized GPU UUID required')


def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE), 'Every input/output path must stay in authorized phase')
    return path


def receipt(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'Exact custody receipt required')
    path = Path(item['path']).resolve()
    require(path.is_relative_to(REPO), 'Custody files must stay in authorized allocation repository')
    require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Pinned bytes differ: ' + str(path))
    return path


def describe(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'Descriptors stay in authorized allocation repository')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def gate(config_path, admission_path, stage):
    route()  # Before source/config/project reads.
    here = scoped(Path(__file__).parent)
    seal = read(here / 'SEAL.json')
    for name, digest in seal['files'].items():
        require(sha(here / name) == digest, 'Integrated source seal changed')
    config_path, admission_path = scoped(config_path), scoped(admission_path)
    cfg, admission = read(config_path), read(admission_path)
    require(admission.get('stage') == stage and admission.get('admitted') is True
            and admission.get('config_sha256') == sha(config_path)
            and admission.get('integrated_source_seal_sha256') == sha(here / 'SEAL.json')
            and admission.get('source_and_provider_custody_complete') is True,
            'Separate root admission for exact stage/config/source required')
    require(cfg.get('proposal_sha256') == sha(here / 'CONFIG_PROPOSAL.json')
            and cfg.get('seeds') == list(SEEDS) and cfg.get('arms') == list(ARMS)
            and cfg.get('base_seed_by_block') == [10000*b for b in SEEDS]
            and cfg.get('seed_amendment_sha256') == sha(here / 'SEED_AMENDMENT.json')
            and cfg.get('inference_adoption_sha256') == sha(here / 'INFERENCE_ADOPTION.json')
            and cfg.get('dataset') == 'Penn94' and cfg.get('node_count') == 41554
            and cfg.get('feature_dim') == 4814 and cfg.get('edge_entries') == 2724458
            and cfg.get('role_counts') == {'train':19407,'select':4852,'cal':4851,'confirm':9705},
            'Root must freeze exact ten-seed seven-arm complete-graph proposal, no grid')
    require(cfg.get('roles_disjoint_and_complete') is True and cfg.get('known_project_history_eligible') is True
            and cfg.get('public_target_column_removed') is True
            and cfg.get('LINKX_constructor_reset_law') == 'paired_block_seed_once_constructor_then_one_native_reset'
            and cfg.get('binary_metric') == 'raw_accuracy' and cfg.get('CONFIRM_truth_in_model_bundle') is False,
            'Root role/eligibility/encoding/binary scorer/LINKX reset authority required')
    require(cfg.get('device') == 'cuda:0' and 0 < cfg.get('owner_seconds', {}).get(stage, 0) <= {'qualify': 1800, 'acquire': 86400, 'calibrate': 1800, 'confirm': 1800}[stage],
            'Finite allocation-stage owner budget required')
    for item in cfg['source_receipts'].values():
        receipt(item)
    require({'torch','numpy','torch_geometric','torch_geometric.nn.conv.sage_conv',
        'torch_geometric.nn.dense.linear','torch_sparse','scipy','scipy.stats._continuous_distns',
        'scipy.stats._distn_infrastructure','scipy.special._ufuncs'}
        <= set(cfg['provider_modules']), 'Native and pinned t CDF/quantile module bindings required')
    for item in cfg['provider_modules'].values(): receipt(item)
    for item in cfg['provider_receipts']:
        receipt(item)
    if stage in ('qualify', 'acquire'):
        require(admission.get('CONFIRM_truth_access') is False, 'Model acquisition has no confirmation truth')
        require(admission.get('input_custody_receipt_sha256') == sha(receipt(cfg['input_custody_receipt'])),
                'Decoded canonical and proper-role custody must be admitted')
        if stage == 'acquire':
            closed_owner(admission['qualification_owner_receipt'])
            qualification = read(receipt(admission['qualification_receipt']))
            require(qualification['passed'] is True and qualification['config_sha256'] == sha(config_path)
                    and qualification['CONFIRM_truth_access'] is False, 'Actual full-graph qualification required')
    return cfg, admission, {'config_sha256': sha(config_path), 'admission_sha256': sha(admission_path),
        'integrated_source_seal_sha256': sha(here / 'SEAL.json'), 'source_receipts': cfg['source_receipts'],
        'provider_receipts': cfg['provider_receipts']}


def close_native(path):
    value = read(receipt(path))
    require(value['terminal'] is True and value['banks'] == 70 and value['body_fit_units'] == 100
            and len(value['records']) == 70 and value['CONFIRM_truth_access'] is False,
            'All70 banks/100 body records terminal before any CAL or CONFIRM role read')
    require({(r['seed'], r['arm']) for r in value['records']} == {(s, a) for s in SEEDS for a in ARMS},
            'Exact native terminal roster; no surviving-seed promotion')
    for row in value['records']:
        require(len(row['fits']) == (4 if row['arm'] == ARMS[5] else 1)
                and all(u['status'] in ('selected_finite', 'failed_retained') for u in row['fits']),
                'Every declared optimizer-body attempt has one retained terminal record')
        if row['status'] == 'selected_finite':
            receipt(row['selected_logits'])
        for fit in row['fits']:
            if fit['status'] == 'selected_finite':
                receipt(fit['checkpoint'])  # Hash/size only; readers never deserialize native states.
    return value


def closed_owner(item):
    value = read(receipt(item))
    require(value['exit_code'] == 0 and value['timed_out'] is False
        and all(value[k] is True for k in ('direct_child_wait', 'child_pid_absent', 'owned_cuda_pid_absent', 'stage_terminal')),
        'Finite direct child owner cleanup and complete stage receipt required')
    return value
