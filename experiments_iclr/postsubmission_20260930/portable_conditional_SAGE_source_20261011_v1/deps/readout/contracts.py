"""Portable complete Physics readout contract; stdlib-only."""
import hashlib
import json
from pathlib import Path
BLOCKS = (8101,8219,8363,8501,8663,8819,9001,9199,9403,9623)
ARMS = ('BASE','E4','N4','equipped_M1','equipped_I4','published_mean_SAGE_source_port_M1')
MEMBERS = dict(zip(ARMS,(4,4,4,1,4,1)))
N,C = 34493,5
COUNTS = {'CAL':150,'CONFIRM':34093}
def gate(*args,**kwargs):
    raise ValueError("Use the portable complete score CLI, after local closure")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):
            digest.update(part)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    require(not path.exists(),'Fresh complete record only; no overwrite/retry')
    temporary = path.with_suffix(path.suffix+'.tmp')
    require(not temporary.exists(),'Stale partial write retained')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def receipt(item):
    require(isinstance(item,dict) and set(item)=={'path','bytes','sha256'},'Exact path/bytes/SHA-256 receipt')
    path = Path(item['path']).resolve()
    require(path.is_file() and type(item['bytes']) is int and path.stat().st_size==item['bytes']
            and sha(path)==item['sha256'],'Pinned bytes changed: '+str(path))
    return path

def scoped(path):
    return Path(path).resolve()

def close_predictions(item, protocol_sha256):
    value = read(receipt(item))
    return validate_predictions(value,protocol_sha256)

def validate_predictions(value, protocol_sha256):
    require(value.get('schema')=='Physics_selected_predictions_complete_v1' and value.get('terminal') is True
            and value.get('blocks')==list(BLOCKS) and value.get('arms')==list(ARMS)
            and value.get('banks')==60 and value.get('body_fit_units')==90
            and value.get('protocol_sha256')==protocol_sha256
            and value.get('CAL_truth_access') is False and value.get('TEST_truth_access') is False,
            'All selected-state/prediction-bank roster closes before target access')
    rows = value.get('records')
    require(isinstance(rows,list) and len(rows)==60 and {(r['block'],r['arm']) for r in rows}
            =={(b,a) for b in BLOCKS for a in ARMS},'Exact60 banks; no survivor subset')
    bodies = 0
    for row in rows:
        require(row.get('status') in ('selected_finite','failed_retained') and row.get('members')==MEMBERS[row['arm']],
                'Terminal bank and fixed member count')
        fits = row.get('fits'); expected=4 if row['arm']=='equipped_I4' else 1
        require(isinstance(fits,list) and len(fits)==expected,'All90 body attempts retained')
        bodies += len(fits)
        for fit in fits:
            require(fit.get('status') in ('selected_finite','failed_retained'),'Terminal body state only')
            if fit['status']=='selected_finite':
                receipt(fit['selected_state_receipt'])  # Hash only; never deserialize.
        if row['status']=='selected_finite':
            require(all(f['status']=='selected_finite' for f in fits),'Finite bank requires every body')
            receipt(row['selected_CAL_archive']);receipt(row['selected_CONFIRM_archive'])
    require(bodies==90 and value.get('local_acquisition_and_export_complete') is True,
            'Complete90 selected bodies and local acquisition/export closure')
    require(isinstance(value.get('cost_records'),list) and isinstance(value.get('source_bindings'),dict),
            'Local run retains full preparation/acquisition/failed-attempt/export cost and source custody records')
    return value
