"""Numerical-only child; observe Torch peaks without changing scientific code."""
import argparse
from datetime import datetime,timezone
import json,os,runpy,sys,threading,time
from pathlib import Path


def write(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-release",required=True,type=Path)
    parser.add_argument("--supervision-output",required=True,type=Path)
    args=parser.parse_args()
    release=json.loads(args.root_release.read_text())
    inv=release["authorized_invocations"]
    if release["authorized_stages"]!=["numerical"] or len(inv)!=1 or inv[0]["stage"]!="numerical" or inv[0]["unit"]!="pair" or inv[0]["base_seed"]!=0:
        raise RuntimeError("This child supports only the reviewed numerical pair")
    caps=release["numerical_caps"]
    driver=Path(release["numerical_driver_path"]).resolve()
    expected=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/graph_ncNC_structural_pattern_pilot_preparation_20261003_v2/pattern_run.py')
    if driver!=expected:raise RuntimeError("Wrong sealed numerical driver")
    stopped=threading.Event();state={"schema":"ncnc-pattern-numerical-child-CUDA-peaks-v1","PID":os.getpid(),"CUDA_observed":False,
        "cuda_peak_allocated_bytes":0,"cuda_peak_reserved_bytes":0,"caps":caps,"source_reset_or_model_hooks":False}
    path=args.supervision_output/"CHILD_CUDA_PEAKS.json"
    def sample():
        module=sys.modules.get("torch")
        cuda=getattr(module,"cuda",None)
        if cuda is None or not hasattr(cuda,"is_initialized") or not cuda.is_initialized():return
        state["CUDA_observed"]=True
        for name,function in (("cuda_peak_allocated_bytes",cuda.max_memory_allocated),("cuda_peak_reserved_bytes",cuda.max_memory_reserved)):
            state[name]=max(state[name],int(function(0)))
        state["UTC"]=datetime.now(timezone.utc).isoformat()
        exceeded=[name for name in ("cuda_peak_allocated_bytes","cuda_peak_reserved_bytes") if state[name]>caps[name]]
        if exceeded:state["cap_violation"]={"kind":"CUDA_peak","fields":exceeded,"exit_code":88}
        write(path,state)
        if exceeded:os._exit(88)  # only this numerical child; supervisor records exit
    def observe():
        while not stopped.wait(.25):
            try:sample()
            except Exception as error:
                state["monitor_failure"]={"type":type(error).__name__,"condition":str(error)}
                write(path,state);os._exit(89)
    write(path,state)
    observer=threading.Thread(target=observe,name="own-numerical-CUDA-peak-observer",daemon=True)
    argv=[str(driver),"--root-release",str(args.root_release.resolve()),"--stage","numerical","--unit","pair","--base-seed","0","--output",inv[0]["output_directory"]]
    state["driver_argv"]=argv
    observer.start()
    try:
        sys.path.insert(0,str(driver.parent));sys.argv=argv
        runpy.run_path(str(driver),run_name="__main__")
    finally:
        stopped.set();observer.join();sample();write(path,state)
    if not state["CUDA_observed"]:raise RuntimeError("Successful numerical stage had no CUDA peak observation")


if __name__=="__main__":main()
