"""Read exact component receipts; preserve reset-aware collection behaviour."""
from pathlib import Path
import importlib.util

HERE=Path(__file__).resolve().parent
previous=HERE.parent/'graph_view_full_shape_qualification_root_20261004_v1/collect_after_transport_reset.py'
spec=importlib.util.spec_from_file_location('retained_qa_collector_v1',previous)
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
client.HERE=HERE

if __name__=='__main__':client.main()
