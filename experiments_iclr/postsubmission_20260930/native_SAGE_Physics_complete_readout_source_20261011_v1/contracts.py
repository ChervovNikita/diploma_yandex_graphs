"""Physics readout closure/admission checks; no model or array imports."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCKS = (8101,8219,8363,8501,8663,8819,9001,9199,9403,9623)
ARMS = ('BASE','E4','N4','equipped_M1','equipped_I4','published_mean_SAGE_source_port_M1')
MEMBERS = dict(zip(ARMS,(4,4,4,1,4,1)))
N, C = 34493, 5
COUNTS = {'CAL':150,'CONFIRM':34093}

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
    require(path.is_relative_to(REPO),'Readout custody stays in authorized scientific repository')
    require(path.is_file() and type(item['bytes']) is int and path.stat().st_size==item['bytes']
            and sha(path)==item['sha256'],'Pinned bytes changed: '+str(path))
    return path

def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE),'Output/config/admission stays in authorized research phase')
    return path

def route():
    require(socket.gethostname()=='anogena-2-0','Literal scientific hostname required before project reads')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID],
            'Sole authorized GPU UUID required; reader itself is CPU only')

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
    require(bodies==90 and value.get('all_native_owners_closed_and_clean') is True,
            'Complete90 selected bodies and root owner closure')
    require(isinstance(value.get('cost_records'),list) and isinstance(value.get('source_bindings'),dict),
            'Root retains full preparation/acquisition/failed-attempt/export cost and source custody records')
    return value

def gate(config_path, admission_path, stage):
    route()
    here = scoped(Path(__file__).parent)
    seal = read(here/'SEAL.json')
    for name,digest in seal['files'].items():
        require(sha(here/name)==digest,'Readout source seal changed: '+name)
    cfg_path,adm_path = scoped(config_path),scoped(admission_path)
    cfg,adm = read(cfg_path),read(adm_path)
    protocol_sha = sha(here/'PROTOCOL.json')
    require(cfg.get('enabled') is True and cfg.get('blocks')==list(BLOCKS) and cfg.get('arms')==list(ARMS)
            and cfg.get('protocol_sha256')==protocol_sha and cfg.get('node_count')==N
            and cfg.get('classes')==C and cfg.get('role_counts')==COUNTS,
            'Root binds fixed Physics protocol/config; default source template disabled')
    require(adm.get('admitted') is True and adm.get('stage')==stage and adm.get('config_sha256')==sha(cfg_path)
            and adm.get('source_seal_sha256')==sha(here/'SEAL.json')
            and adm.get('prediction_closure')==cfg.get('prediction_closure')
            and adm.get('all60_banks_90_bodies_closed') is True,
            'Separate exact root stage admission required')
    require(set(cfg.get('roles',{}))=={str(b) for b in BLOCKS}
            and all(set(v)=={'cal_ids','confirm_ids','cal_truth','confirm_truth'} for v in cfg['roles'].values()),
            'Ten fixed per-block role receipts; target files opened only in their stage')
    require(cfg.get('proper_roles_admitted') is True and cfg.get('roles_disjoint_and_complete') is True
            and cfg.get('TEST_truth_in_predictions') is False,'Actual root class-stratified role/custody admission')
    cap = cfg.get('stage_seconds',{}).get(stage)
    require(type(cap) is int and 0<cap<=3600,'Root supplies finite CPU-stage cap <=3600 seconds')
    required={'numpy','torch','scipy','scipy.stats._continuous_distns','scipy.stats._distn_infrastructure','scipy.special._ufuncs'}
    require(required<=set(cfg.get('provider_modules',{})),'Pinned NumPy/Torch/SciPy inference modules')
    for item in cfg['provider_modules'].values():receipt(item)
    predictions=close_predictions(cfg['prediction_closure'],protocol_sha)
    if stage=='calibrate':
        require(adm.get('CAL_truth_access') is True and adm.get('TEST_truth_access') is False,
                'CAL-only target release after complete prediction closure')
    else:
        require(adm.get('one_complete_TEST_truth_release') is True and adm.get('TEST_truth_access') is True
                and adm.get('no_prior_TEST_score_in_scope') is True,'One root-authorized complete TEST release')
    custody={'config_sha256':sha(cfg_path),'admission_sha256':sha(adm_path),'protocol_sha256':protocol_sha,
             'source_seal_sha256':sha(here/'SEAL.json'),'prediction_closure':cfg['prediction_closure']}
    return cfg,adm,custody,predictions
