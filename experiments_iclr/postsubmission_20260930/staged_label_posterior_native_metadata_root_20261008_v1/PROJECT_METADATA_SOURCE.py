from pathlib import Path
import json,socket,hashlib
import torch
assert socket.gethostname()=='anogena-2-0'
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
ROWS=[{'seed': 6101, 'path': 'label_only_four_bank_first_screen_execution_root_20261008_v2/seed6101/NATIVE_OWN_BEST_ALL_BANKS.pt', 'bytes': 99249338, 'sha256': '3be5b9408b7882e1941ee4884b8bdde5c0a94b97f0ef94ad8e6a4c1a7b051b9c'}, {'seed': 6203, 'path': 'label_only_four_bank_first_screen_execution_root_20261008_v2/seed6203/NATIVE_OWN_BEST_ALL_BANKS.pt', 'bytes': 99249338, 'sha256': '7350628888ea60c8a3dba31aeadb5b3720e88bcc21571f59eb73781227aede54'}, {'seed': 6307, 'path': 'label_only_four_bank_first_screen_execution_root_20261008_v2/seed6307/NATIVE_OWN_BEST_ALL_BANKS.pt', 'bytes': 99249338, 'sha256': 'c03d2bd7044c64596b1a1efcf27bee693a1923ebd6891ec9e38b562f552ff818'}];out=[]
for row in ROWS:
 f=P/row['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'] and f.stat().st_size==row['bytes'];v=torch.load(f,map_location='cpu',weights_only=False)
 assert v['kind']=='native_own_best' and v['run']['seed']==row['seed']
 out.append(dict(**row,schema=v['schema'],kind=v['kind'],arm=v['arm'],epoch=v['epoch'],global_mode=v['global_mode'],snapshot_purpose=v['snapshot_purpose'],selector_performed=v['selector_performed'],native_checkpoint_kind=v['native']['checkpoint_kind'],run_identity=v['run']))
print(json.dumps(dict(native_states=out,weights_or_outcome_values_exported=False,method='Metadata projection of already-opened trusted own-best snapshots after whole family closure, no native construction/forward.'),allow_nan=False))
