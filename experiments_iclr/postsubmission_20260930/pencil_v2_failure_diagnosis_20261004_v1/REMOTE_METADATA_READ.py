from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'pencil_collab_paired_predictive_execution_root_20261004_v2'
source=phase/'pencil_collab_paired_predictive_preparation_20261004_v2'
assert Path.cwd()==repo and os.uname().nodename=='peptide'
allowed=('seed0/supervision/run01/PHYSICAL_TERMINAL.json','seed0/supervision/run01/STATUS.json','seed0/supervision/run01/STARTED.json','seed0/run01/MONITOR_FAILURE.json','QUEUE_FAILURE.json','QUEUE_EXCEPTION.json')
receipts=[]
for relative in allowed:
 path=root/relative
 assert path.resolve().is_relative_to(root)
 if not path.exists():
  receipts.append(dict(path=relative,present=False));continue
 assert path.is_file() and not path.is_symlink() and path.stat().st_size<2000000
 raw=path.read_bytes()
 value=json.loads(raw)
 receipts.append(dict(path=relative,present=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata=value))
source_digests=[]
for name in ('MANIFEST.json','SEAL.json','supervise.py','worker.py','common.py'):
 path=source/name
 assert path.is_file() and not path.is_symlink()
 raw=path.read_bytes()
 source_digests.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='SCOPED_FAILURE_RESOURCE_METADATA_ONLY',root=str(root),receipts=receipts,source_digests=source_digests,predictive_values_read=False,selected_checkpoint_reads=False,data_reads=False,signals_sent=False,launches=False,automatic_retry=False)))
