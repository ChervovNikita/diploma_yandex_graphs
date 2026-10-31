"""Fresh-worker facade for the explicit campaign host authority."""
import importlib.util
from pathlib import Path
import sys
path=Path(__file__).resolve().parent.parent/'host_binding.py'
name='_server51_composed_host_authority'
if name not in sys.modules:
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
for key in ('guard','bound_project_root','validate_host_job','load_config','require','normalized_absolute'):
    globals()[key]=getattr(sys.modules[name],key)
