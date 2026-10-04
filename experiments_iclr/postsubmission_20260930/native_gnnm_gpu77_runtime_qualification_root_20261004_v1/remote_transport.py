from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
P=HERE.parent
spec=importlib.util.spec_from_file_location('retained_graph_view_gpu77_transport',P/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.HERE=HERE
run=module.run
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
