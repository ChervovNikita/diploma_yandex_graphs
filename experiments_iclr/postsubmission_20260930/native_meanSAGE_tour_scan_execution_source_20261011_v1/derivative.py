"""One admitted CPUfloat64 derivative check; no data or model construction."""
from pathlib import Path
import resource
import time
from runtime import load,require,write

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();result=dict(complete=False,custody=custody,source_manifest=config['source_manifest'],data_access=False,models_constructed=0,retry=False)
    try:
        import torch
        require(torch.__version__==config['expected_runtime_versions']['torch'],'Bound derivative runtime version')
        torch.set_num_threads(2)
        result.update(load('gradient_check').run(torch,load('associative_scan')))
        require(time.monotonic()-start<config['limits']['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=config['limits']['rss_bytes'],'Finite short derivative wall/RSS cap')
    except BaseException as e:
        result.update(complete=False,failure=dict(type=type(e).__name__,message=str(e)));raise
    finally:
        result['seconds']=time.monotonic()-start;result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write(output/'DERIVATIVE.json',result)
    return result
