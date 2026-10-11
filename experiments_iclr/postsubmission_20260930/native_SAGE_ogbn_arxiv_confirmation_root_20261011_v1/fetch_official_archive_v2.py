"""Retry the exact provider's published HTTP route after TLS timeout; preserve V1."""
from datetime import datetime, timezone
import base64
import importlib.util
import json
from pathlib import Path
import shlex
import zlib
import fetch_official_archive_v1 as prior

HERE=Path(__file__).resolve().parent
P=HERE.parent

def main():
    canonical='http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
    payload=dict(canonical_url=canonical,requested_url=canonical,
                 provider_revision='61e9784ca76edeaa6e259ba0f836099608ff0586')
    (HERE/'OFFICIAL_ARCHIVE_BYTE_ADMISSION_V2.json').write_text(json.dumps(dict(
        UTC=datetime.now(timezone.utc).isoformat(),operation='Exact published official HTTP release after V1 TLS handshake timeout',
        prior_failure_receipt='OFFICIAL_ARCHIVE_BYTE_TRANSPORT_V1.json',
        no_data_decoding_or_fit_admission=True,**payload),indent=2)+'\n')
    remote=prior.REMOTE.replace('OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V1.json','OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V2.json').replace('https_transport_amendment=True','https_transport_amendment=False')
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
          '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
          shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,prior.REPO,prior.UUID,'official-arxiv-bytes-only'])]
    r=mod.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(),capture_output=True,text=True,timeout=350)
    (HERE/'OFFICIAL_ARCHIVE_BYTE_TRANSPORT_V2.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stdout[-2500:]
    receipt=json.loads(r.stdout);assert receipt['complete']
    (HERE/'OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V2.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('complete','UTC','bytes','sha256','seconds','model_fits','scientific_fit_admitted')}))

if __name__=='__main__':main()
