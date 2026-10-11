"""Bind one fixed after-fit continuation; no arrays, checkpoints or scores."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = Path(__file__).resolve().parent
S = P / 'native_SAGE_Physics_after_fit_pipeline_source_20261011_v1'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def item(path):
    p = Path(path).resolve(); assert p.is_relative_to(R)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def write(name, value):
    q = H / name; assert not q.exists(); q.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == R
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID]
    seal = json.loads((S / 'SEAL.json').read_text())
    for name, digest in seal['files'].items(): assert item(S / name)['sha256'] == digest
    cfg = json.loads((S / 'CONFIG_TEMPLATE.json').read_text()); cfg['enabled'] = True
    cfg['cohort_owner_start'] = item(H / 'cohort_owner_v1/OWNER_START.json')
    cfg['prepared_receipt'] = item(H / 'prepared_v1/CUSTODIAL_EXPORT_RECEIPT.json')
    cfg['export_limits'] = dict(wall_seconds=3600, cuda_bytes=48 * 1024**3, rss_bytes=64 * 1024**3)
    # Numerical packages are imported only to pin their code; no data is read.
    cfg['provider_modules'] = {name: item(importlib.import_module(name).__file__) for name in cfg['provider_modules']}
    cfg['prior_cost_receipts'] = [item(H / name) for name in ('PROVIDER_BYTE_RECEIPT_V1.json',
        'CUSTODIAL_EXPORT_OBSERVATION_V1.json', 'RESOURCES_OBSERVATION_V1.json', 'AUTHOR_RESOURCES_OBSERVATION_V1.json',
        'COMPLETE_COHORT_BIND_TRANSPORT_V1.json', 'COMPLETE_COHORT_BIND_TRANSPORT_V2.json', 'COMPLETE_COHORT_BIND_TRANSPORT_V3.json')]
    cfg['prior_cost_receipts'].append(item(P / 'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1/QUALIFICATION_METADATA_REGISTERED_V1.json'))
    write('AFTER_FIT_PIPELINE_CONFIG_V1.json', cfg)
    admission = dict(UTC=datetime.now(timezone.utc).isoformat(), admitted=True, action='Physics_after_fit_one_shot',
                     config_sha256=item(H / 'AFTER_FIT_PIPELINE_CONFIG_V1.json')['sha256'], pipeline_seal_sha256=item(S / 'SEAL.json')['sha256'],
                     cohort_owner_start=cfg['cohort_owner_start'], all_phases_pre_authorized=True, no_retries_or_recipe_changes=True,
                     scientific_protocol_unchanged=True, banks=60, optimizer_bodies=90, fixed_CAL_endpoints=60,
                     one_complete_TEST_release=True, no_current_heldout_quality_access=True,
                     extra_GNN_fit_updates=0, paper_review_admitted=False,
                     admission_basis='Root authorizes the frozen full-roster export, calibration and scorer after verified complete acquisition; no outcome-dependent decision')
    write('AFTER_FIT_PIPELINE_ADMISSION_V1.json', admission)
    print(json.dumps(dict(complete=True, config=cfg, admission=admission, config_receipt=item(H / 'AFTER_FIT_PIPELINE_CONFIG_V1.json'),
                         admission_receipt=item(H / 'AFTER_FIT_PIPELINE_ADMISSION_V1.json'), scientific_updates=0, CAL_TEST_scoring=False)))


if __name__ == '__main__': main()
