"""Recover four small pinned paper dependencies from the existing server archive."""
import ast
import base64
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent
P=H.parent

def main():
    request=json.loads((H/'PINNED_ASSET_REQUEST_V1.json').read_text())
    names={r['local_name'] for r in request['objects']}
    assert names=={'architecture_sharing_compact.png','gnnm_vs_ens_heatmap.png','iclr2027_conference.sty','iclr2027_conference.bst'}
    assert request['maximum_payload_bytes']==376966
    # Reuse the already guarded streaming archive reader, without restoring
    # the old evidence tree or loading arrays/checkpoints/reviews.
    parent=P/'fresh_manuscript_review_preparation_20261011_v1/fetch_historical_source_v1.py'
    tree=ast.parse(parent.read_text())
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets)]
    assert len(assignments)==1
    remote=ast.literal_eval(assignments[0].value)
    spec=importlib.util.spec_from_file_location('pinned_archive_transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-p','2222',
        'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,
        '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs','GPU-44039938-fd82-41d2-fefd-de71514e2fac','pinned-paper-assets'])]
    result=mod.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps({'objects':request['objects']}).encode())).decode(),capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stderr[-1500:]
    obj=json.loads(result.stdout);assert obj['complete'] and set(obj['objects'])==names
    target=H/'historical_round18_assets';target.mkdir(exist_ok=False)
    for name,row in obj['objects'].items():
        b=base64.b64decode(row.pop('base64'));assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
        (target/name).write_bytes(b)
    obj.update(UTC=datetime.now(timezone.utc).isoformat(),request_sha256=hashlib.sha256((H/'PINNED_ASSET_REQUEST_V1.json').read_bytes()).hexdigest(),
        source_scope='Four historical round18 paper dependencies only; not asserted to be the final submitted-source version',
        no_Desktop_access=True,scores_changed=False,compiled=False,large_archive_restored=False)
    (H/'ASSET_RECOVERY_RECEIPT_V1.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({k:v for k,v in obj.items() if k!='objects'}));print('recovered small files',sorted(names))

if __name__=='__main__':main()
