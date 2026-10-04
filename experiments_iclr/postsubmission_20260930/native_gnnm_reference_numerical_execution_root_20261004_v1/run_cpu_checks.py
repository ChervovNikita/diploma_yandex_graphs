"""Exact ordinary-native reference objective/state tests; synthetic CPU only."""
from pathlib import Path
import hashlib
import importlib.util

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
previous=PHASE/'graph_view_source_numerical_execution_root_20261004_v1/run_cpu_checks.py'
spec=importlib.util.spec_from_file_location('retained_bank_cpu_transport_v1',previous)
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
client.HERE=HERE
client.SOURCE=PHASE/'accuracy_first_native_gnnm_reference_source_preparation_20261004_v1'
client.MANIFEST='06aa22d011b67c186921480a653c2d8060248d255a87a9b39b3455610216c6cb'
client.PROTOCOL=hashlib.sha256((client.SOURCE/'PROTOCOL.json').read_bytes()).hexdigest()
client.REMOTE=client.REMOTE.replace('graph_view_source_numerical_execution_root_20261004_v1',HERE.name).replace(
    'accuracy_first_graph_view_source_preparation_20261004_v1',client.SOURCE.name)
client.REMOTE=client.REMOTE.replace("PYTHONDONTWRITEBYTECODE='1')", "PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root))")

if __name__=='__main__':client.main()
