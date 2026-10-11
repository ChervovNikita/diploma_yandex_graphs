"""Bind source, acquired bytes and runtime for custodial export only."""
from datetime import datetime,timezone
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'native_CoauthorPhysics_custodial_provider_source_20261011_v1'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def item(path):
    q=Path(path).resolve();assert q.is_relative_to(R)
    d=hashlib.sha256()
    with q.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return dict(path=str(q),bytes=q.stat().st_size,sha256=d.hexdigest())
def write(p,r):assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    assert Path.cwd()==R
    for row in json.loads((S/'MANIFEST.json').read_text())['files']:
        assert item(S/row['path'])['sha256']==row['sha256']
    protocol=json.loads((H/'PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json').read_text())
    assert protocol['banks']==60 and protocol['optimizer_bodies']==90 and not protocol['scientific_fit_admitted']
    write(H/'ROOT_HISTORY_AND_ACCESS_DECISION_V1.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
        dataset='CoauthorPhysics',decision='Prospective confirmation within the audited authored research account; no prior evaluation found, global absence not certified',
        selection_before_new_arrays_or_targets=True,root_protocol=item(H/'PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json'),
        earlier_candidates='Arxiv encountered results excluded; GitHub fixed release unavailable, no substitute feature release used',
        scope='All relevant permitted authored notes and saved source-only Physics reservations; no unauthorized allocation access',
        scientific_fit_admitted=False,TEST_scoring_admitted=False))
    write(H/'CUSTODIAL_EXPORT_ADMISSION_V1.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
        action='Custodial provider and ten paired role exports only',source=item(S/'export_physics.py'),
        all_targets_including_TEST_parsed_by_custodian=True,predictive_outputs_or_scores=False,
        model_labels_only_TRAIN=True,SELECT_separate=True,CAL_and_TEST_separate=True,
        scientific_fit_admitted=False,resource_qualification_admitted=False,TEST_scoring_admitted=False,
        source_review='Root inspected actual exporter, pinned input/count/support/role contracts; no manuscript reviewer or tiny surrogate run',
        finite_owner_seconds=1200,normal_execution=True))
    cfg=json.loads((S/'CONFIG_TEMPLATE.json').read_text());cfg['admission']['custodial_export']=True
    cfg['output_directory']=str(H/'prepared_v1')
    cfg['input_npz']=item(H/'raw/ms_academic_phy.npz')
    cfg['input_preparation_admission_receipt']=item(H/'CUSTODIAL_EXPORT_ADMISSION_V1.json')
    cfg['root_history_and_access_decision_receipt']=item(H/'ROOT_HISTORY_AND_ACCESS_DECISION_V1.json')
    cfg['byte_acquisition_receipt']=item(H/'PROVIDER_BYTE_RECEIPT_V1.json')
    cfg['source_receipts']=[item(S/'pinned_sources'/n) for n in ('PyG_Coauthor.py','PyG_read_npz.py','PRIMARY_SCOPES.json')]
    names=('numpy','scipy','scipy.sparse','scipy.sparse.csgraph','torch','torch_geometric','torch_geometric.utils.loop','torch_geometric.utils.undirected')
    modules={n:importlib.import_module(n) for n in names}
    cfg['runtime_source_receipts']=[item(m.__file__) for m in modules.values()]
    cfg['expected_runtime_versions']={n:modules[n].__version__ for n in ('numpy','scipy','torch','torch_geometric')}
    write(H/'CUSTODIAL_EXPORT_CONFIG_V1.json',cfg)
    print(json.dumps(dict(complete=True,config=item(H/'CUSTODIAL_EXPORT_CONFIG_V1.json'),runtime=cfg['expected_runtime_versions'],scientific_fits=0,TEST_scoring=False)))

if __name__=='__main__':main()
