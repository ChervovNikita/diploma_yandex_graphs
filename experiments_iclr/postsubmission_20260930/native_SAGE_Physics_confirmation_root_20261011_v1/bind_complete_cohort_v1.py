"""Freeze complete fixed Physics acquisition after the two actual resource PASS."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = Path(__file__).resolve().parent
SEEDS = [8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623]
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def item(path):
    p = Path(path).resolve(); assert p.is_relative_to(R)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def write(name, value):
    path = H / name; assert not path.exists()
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == R
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID]
    S = P / 'native_SAGE_sequential_own_CE_Physics_source_20261011_v5'
    E = P / 'native_SAGE_endpoint_value_transport_source_20261011_v2'
    C = P / 'native_CoauthorPhysics_author_meanSAGE_context_source_20261011_v1'
    I = P / 'native_SAGE_Physics_coupled_fit_integration_source_20261011_v1'
    qualifications = [P / 'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1/engine_qualification_v3/ENGINE_EQUIVALENCE.json',
                      H / 'resources_qualification_v1/PHYSICS_RESOURCE_QUALIFICATION.json',
                      H / 'author_resources_qualification_v1/AUTHOR_MEANSAGE_RESOURCE_QUALIFICATION.json']
    for path in qualifications: assert json.loads(path.read_text())['status'] == 'PASS'
    owners = [H / 'resources_owner_v1/OWNER_END.json', H / 'author_resource_owner_v1/OWNER_END.json']
    for path in owners:
        v = json.loads(path.read_text()); assert v['status'] == 'COMPLETE_EXIT_0' and v['exit_code'] == 0 and v['child_pid_absent']
    prepared = H / 'prepared_v1'
    exporter = json.loads((prepared / 'CUSTODIAL_EXPORT_RECEIPT.json').read_text()); assert exporter['complete']
    cfg = json.loads((H / 'RESOURCES_CONFIG_V1.json').read_text()); cfg.pop('root_admission', None)
    cfg.pop('label_free_SELECT_ids', None)
    cfg['family_limits'] = dict(wall_seconds=172800, cuda_bytes=48 * 1024**3, rss_bytes=64 * 1024**3)
    assert cfg['seeds'] == SEEDS and cfg['profile'] == exporter['profiles']['8101']
    write('CORE_CONFIG_V1.json', cfg)
    model_spec = importlib.util.spec_from_file_location('Physics_source_point_metadata', C / 'author_model.py')
    m = importlib.util.module_from_spec(model_spec); sys.modules[model_spec.name] = m; model_spec.loader.exec_module(m)
    author = dict(execution_enabled=True, device='cuda:0', seeds=SEEDS, point=m.POINT,
                  profile=dict(dataset='CoauthorPhysics', nodes=34493, input_dim=8415, classes=5, edges=530417, self_edges=34493,
                               train_rows=100, select_rows=150, cal_rows=150, confirm_rows=34093),
                  limits=dict(wall_seconds=172800, cuda_bytes=8 * 1024**3, rss_bytes=16 * 1024**3),
                  role_blocks={str(seed): dict(train=str(prepared / f'block_{seed}/train.npz'), select=str(prepared / f'block_{seed}/select.npz'),
                                              cal_ids=str(prepared / f'block_{seed}/cal_ids.npy'), confirm_ids=str(prepared / f'block_{seed}/confirm_ids.npy')) for seed in SEEDS})
    write('AUTHOR_CONFIG_V1.json', author)
    bound = {}
    def add(path):
        q = item(path); bound[q['path']] = q
    for folder in (S, E, C, I):
        manifest = json.loads((folder / 'MANIFEST.json').read_text())
        for row in manifest['files']:
            q = item(folder / row['path']); assert q['sha256'] == row['sha256']; bound[q['path']] = q
        for name in ('MANIFEST.json', 'SEAL.json'): add(folder / name)
    for q in cfg['source_receipts'].values(): assert item(q['path']) == q; bound[q['path']] = q
    for key in ('common_routes', 'factors'): add(cfg[key])
    for block in author['role_blocks'].values():
        for path in block.values(): add(path)
    for seed in SEEDS:
        for role in ('train_ids', 'select_ids'): add(prepared / f'block_{seed}/{role}.npy')
    for name in ('PROVIDER_IDENTITY.json', 'CUSTODIAL_EXPORT_RECEIPT.json', 'provider_support.npz'): add(prepared / name)
    for name in ('PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json', 'bind_complete_cohort_v1.py', 'complete_cohort_outer_owner_v1.py'): add(H / name)
    add(P / 'native_SAGE_endpoint_value_execution_root_20261011_v1/RUNTIME_FINGERPRINT_V1.json')
    for path in qualifications + owners: add(path)
    for name in ('CORE_CONFIG_V1.json', 'AUTHOR_CONFIG_V1.json'): add(H / name)
    qual = json.loads(qualifications[1].read_text())
    release = dict(status='ROOT_CONFIRMATION_FIT_ADMITTED', capable_reference_decision='ROOT_ADMITTED',
                   engine_equivalence='ROOT_VERIFIED_PASS', actual_physics_resource_qualification='ROOT_VERIFIED_PASS')
    admission = dict(UTC=datetime.now(timezone.utc).isoformat(), status='ROOT_COMPLETE_PHYSICS_COHORT_ADMITTED', banks=60, optimizer_bodies=90,
                     bound_files=list(bound.values()), qualification_receipts=[item(q) for q in qualifications], qualification_owners=[item(q) for q in owners],
                     core_config=item(H / 'CORE_CONFIG_V1.json'), author_config=item(H / 'AUTHOR_CONFIG_V1.json'),
                     protocol=item(H / 'PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json'), coupled_runner=item(I / 'run_coupled.py'),
                     core_port_directory=str(S), e_directory=str(E), author_context_directory=str(C), core_release=release,
                     expected_port_receipt=qual['port_receipt'],
                     author_execution_sources={name: item(C / name) for name in ('consumer.py', 'author_model.py')},
                     preserved_scientific_recipe=True, no_current_heldout_quality_access=True,
                     context_alias={'author_meanSAGE_M1': 'published_mean_SAGE_source_port_M1'},
                     primary='Frozen paired ten-mask E-BASE and E-I4 Holm superiority, practical mean screens and separate NLL diagnostics',
                     external_watchdog_seconds=172920, parent_wall_seconds=172800,
                     overlap='Existing unchanged Penn94; observed costs not isolated speed evidence')
    write('FROZEN_BINDINGS_V1.json', admission)
    spec = dict(status='ROOT_COUPLED_FIT_ADMITTED', H=str(H), parent_wall_seconds=172800,
                core_config=admission['core_config'], author_config=admission['author_config'],
                root_hook=item(H / 'ROOT_HOOK_V1.py'), root_bindings=item(H / 'FROZEN_BINDINGS_V1.json'),
                core_port_directory=str(S), e_directory=str(E), author_context_directory=str(C), core_release=release)
    write('ROOT_INTEGRATION_V1.json', spec)
    print(json.dumps(dict(complete=True, admission=admission, spec=spec, spec_receipt=item(H / 'ROOT_INTEGRATION_V1.json'), core_config=cfg, author_config=author)))


if __name__ == '__main__': main()
