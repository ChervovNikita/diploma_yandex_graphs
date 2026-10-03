"""Acquire the full public HGB DBLP archive inside the authorized repository.

This script records archive metadata only. It never opens label payloads or
executes training. The Google Drive identifier comes from the current author
release landing page; the archive hash is measured after retrieval.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import traceback
import urllib.request
import zipfile

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PACKET = REPO/'experiments_iclr/postsubmission_20260930/hgb_data_acquisition_root_v1'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
FILE_ID = '1fLLoy559V7jJaQ_9mQEsC06VKd6Qd3SC'
URL = 'https://drive.usercontent.google.com/download?id='+FILE_ID+'&export=download&authuser=0&confirm=t'


def main():
    assert Path.cwd().resolve() == REPO and REPO.resolve() == REPO
    assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=REPO,
                          capture_output=True,text=True,check=True).stdout.strip() == str(REPO)
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                          capture_output=True,text=True,check=True).stdout.splitlines() == [UUID]
    run = PACKET/'download01'
    run.mkdir(exist_ok=False)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),route_uuid=UUID,
                   source_URL=URL,author_landing_URL='https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing',
                   file_id=FILE_ID,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   native_models_executed=False,label_payloads_opened=False,training_launched=False)
    partial = run/'DBLP.zip.partial'
    try:
        digest = hashlib.sha256()
        total = 0
        request = urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            receipt.update(HTTP_status=response.status,final_URL=response.url,
                           content_type=response.headers.get('Content-Type'),
                           content_disposition=response.headers.get('Content-Disposition'))
            with partial.open('xb') as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > 100_000_000:
                        raise ValueError('Unexpectedly large DBLP archive; inspect release before continuing')
                    handle.write(block)
                    digest.update(block)
        receipt.update(downloaded_bytes=total,archive_sha256=digest.hexdigest())
        if not zipfile.is_zipfile(partial):
            raise ValueError('Response is not a ZIP archive')
        members = []
        with zipfile.ZipFile(partial) as archive:
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                assert not path.is_absolute() and '..' not in path.parts
                members.append(dict(path=item.filename,uncompressed_bytes=item.file_size,
                                    compressed_bytes=item.compress_size,CRC32=item.CRC))
        partial.rename(run/'DBLP.zip')
        receipt.update(status='archive_acquired_metadata_only',archive_path=str(run/'DBLP.zip'),members=members,
                       archive_payloads_parsed=False,declared_landing_size_bytes=2567741,
                       landing_size_matches=(total == 2567741))
    except Exception as error:
        receipt.update(status='acquisition_failed',error_type=type(error).__name__,
                       error_message=str(error),traceback=traceback.format_exc(),
                       retained_partial_bytes=partial.stat().st_size if partial.exists() else 0)
    with (run/'ACQUISITION.json').open('x') as handle:
        json.dump(receipt,handle,indent=2)
        handle.write('\n')
    print(json.dumps(receipt))
    return 0 if receipt['status']=='archive_acquired_metadata_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
