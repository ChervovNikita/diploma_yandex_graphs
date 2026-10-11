"""Run the frozen SAGE fixed graph-key through its reviewed native acquisition lifecycle."""
from pathlib import Path
import hashlib,importlib.util,json,socket,subprocess,sys
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    frozen=json.loads((H/'FREEZE.json').read_text())
    for row in frozen['bound_files']:
        assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    assert json.loads((H/'ACTUAL_QUALIFICATION_V1.json').read_text())['qualified']
    assert json.loads((H/'REFERENCE_ADMISSION.json').read_text())['admitted']
    assert json.loads((H/'CALIBRATION_REUSE_ADMISSION.json').read_text())['admitted']
    output=H/'actual_family_v1';output.mkdir(exist_ok=False)
    source=P/'native_SAGE_fixed_context_key_source_20261011_v1/run_family.py'
    spec=importlib.util.spec_from_file_location('admitted_native_sage_fixed_graph_keys',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=H/'CONFIG.json'
    qualification=json.loads((H/'ACTUAL_QUALIFICATION_V1.json').read_text())
    assert qualification['configuration_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
    family=module.Family(json.loads(config.read_text()),output,hashlib.sha256(config.read_bytes()).hexdigest())
    assert family.context_contract==qualification['target_objective_contract']
    assert family.source_hashes==qualification['source_sha256']
    fields=['array_hashes','actual_loss_tensor_hashes','ordered_TRAIN_ids_int64_bytes_sha256','ordered_TRAIN_y_int64_bytes_sha256','single_order_int64_bytes_sha256','torch_version','numpy_version','target_policy','key_policy','shuffle_seed','teacher_temperature','student_temperature','TRAIN_rows','target_loss_dtype','objective_contract']
    assert all(family.context_preparation[k]==qualification['target_preparation'][k] for k in fields)
    family.run()
if __name__=='__main__':main()
