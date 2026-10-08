from pathlib import Path
import socket,json,hashlib
import torch,numpy as np
assert socket.gethostname()=='anogena-2-0';torch.set_num_threads(1)
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
def digest(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(json.dumps(dict(shape=list(x.shape),dtype=str(x.dtype)),sort_keys=True).encode());h.update(x.numpy().tobytes(order='C'));return h.hexdigest()
roles={}
for name,file,fields in [('train','train.npz',{'x':'TRAIN_x','edge_index':'prepared_edge_index','ids':'TRAIN_ids','y':'TRAIN_y'}),('valid','valid.npz',{'ids':'development_ids','y':'development_y'})]:
 f=P/'label_correction_full_WikiCS_role_projection_allocation_20261008_v1'/file
 owner=json.loads((P/'label_only_four_bank_WikiCS_scientific_owner_source_20261008_v2/SOURCE_BINDINGS.json').read_text());b=owner['roles'][name];assert hashlib.sha256(f.read_bytes()).hexdigest()==b['sha256']
 with np.load(f,allow_pickle=False) as data:
  for field,key in fields.items():roles[key]=digest(torch.from_numpy(data[field].copy()))
print(json.dumps(dict(numeric_role_sha256=roles,metrics_computed=False,role_source_hashes_checked=True,TEST_loaded=False)))
