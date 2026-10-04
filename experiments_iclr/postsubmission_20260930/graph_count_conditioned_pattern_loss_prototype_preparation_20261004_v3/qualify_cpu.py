#!/usr/bin/env python3
"""Externally disabled fabricated CPU qualification; never a scientific runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parent / 'graph_count_conditioned_pattern_cpu_qualification_execution_root_20261004_v3'
CAPS = {'wall_seconds':900,'peak_RSS_bytes':4*1024**3}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def gate(path,pin):
    require(path.resolve()==EXECUTION/'ROOT_RELEASE_fabricated_cpu.json' and not path.is_symlink() and sha(path)==pin,
            'Exact external CPU release required')
    release = json.loads(path.read_text())
    require(release.get('schema')=='graph-count-conditioned-pattern-cpu-root-release-v1' and release.get('status')=='APPROVED'
            and release.get('authorized_stages')==['fabricated_cpu'] and release.get('root_authorization_reference')
            and release.get('scientific_fit_admitted') is False and release.get('GPU_data_access') is False
            and release.get('TEST_supported') is False and release.get('automatic_retry') is False,
            'Fabricated CPU-only root authorization absent')
    require(sha(HERE/'MANIFEST.json')==release['source_manifest_sha256'] and release.get('caps')==CAPS,
            'CPU source/caps differ')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        source = HERE/row['path']
        require(source.resolve().is_relative_to(HERE) and not source.is_symlink() and sha(source)==row['sha256']
                and source.stat().st_size==row['bytes'], 'CPU source payload differs')
    review_path = Path(release['independent_source_review_path'])
    require(review_path.resolve().is_relative_to(HERE.parent) and not review_path.is_symlink()
            and sha(review_path)==release['independent_source_review_sha256'], 'Independent CPU source review absent')
    review = json.loads(review_path.read_text())
    require(review.get('status')=='PASS' and review.get('candidate_manifest_sha256')==release['source_manifest_sha256']
            and review.get('execution_authorized') is False, 'Independent exact source PASS required')
    for row in json.loads((HERE/'SOURCE_BINDING.json').read_text())['external_source_pins']:
        source = HERE.parent/row['path']
        require(source.resolve().is_relative_to(HERE.parent) and not source.is_symlink()
                and sha(source)==row['sha256'] and source.stat().st_size==row['bytes'], 'Sealed algebra/interface changed')
    profile = release['runtime_profile']
    require(Path(sys.executable).resolve()==Path(profile['python_executable']).resolve()
            and sha(Path(profile['python_executable']))==profile['python_executable_sha256']
            and importlib.metadata.version('torch')==profile['torch_distribution_version'], 'Root-pinned CPU runtime differs')
    return release


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release',required=True,type=Path)
    parser.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    release = gate(args.root_release,args.release_sha256)
    output = EXECUTION/'fabricated_cpu/run01'
    require(not output.exists(),'Fresh CPU engineering attempt required; no retry')
    output.mkdir(parents=True)
    ledger = {'comparison_reports':[], 'case_counts':{
        'ESP_pattern_cases':0,'mixture_pattern_cases':0,'single_normalization_groups':0,
        'single_enumerated_patterns':0,'matching_law_patterns':0}}
    last_progress = [started]
    def check_cap():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform=='darwin' else peak*1024)
        require(time.monotonic()-started<=CAPS['wall_seconds'] and peak_bytes<=CAPS['peak_RSS_bytes'], 'CPU wall/RSS cap exceeded')
        if time.monotonic()-last_progress[0] >= 1:
            write(output/'PROGRESS.json',{'case_counts':ledger['case_counts'],
                  'ragged_case_counts':ledger.get('ragged_case_counts'),
                  'vectorized_case_counts':ledger.get('vectorized_case_counts'),
                  'completed_comparison_reports':len(ledger['comparison_reports']),
                  'inclusive_wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak_bytes,
                  'last_comparison_label':ledger['comparison_reports'][-1]['label'] if ledger['comparison_reports'] else None})
            last_progress[0] = time.monotonic()
        return peak_bytes
    try:
        # This process cannot select a CUDA device. No cuda API, data loader,
        # model checkpoint, native graph or optimizer is present in the oracle.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        sys.path.insert(0,str(HERE))
        from oracles import qualify
        result = qualify(check_cap,ledger)
        from ragged_oracles import qualify_ragged
        result['ragged_successor'] = qualify_ragged(check_cap, ledger)
        from vectorized_oracles import qualify_vectorized
        result['vectorized_single_successor'] = qualify_vectorized(check_cap, ledger)
        peak = check_cap()
        write(output/'QUALIFICATION.json',{'schema':'graph-count-conditioned-pattern-fabricated-cpu-qualification-v1',
              'status':'PASS','UTC':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':release['source_manifest_sha256'],
              'root_release_sha256':args.release_sha256,'runtime_profile':release['runtime_profile'],
              'torch_version':torch.__version__,'inclusive_wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak,
              'result':result,'GPU_data_access':False,'scientific_fit_admitted':False,'TEST_supported':False,
              'native_full_batch_resource_qualification':False,'frozen_four_arm_screen_changed':False})
    except BaseException as error:
        write(output/'FAILURE.json',{'status':'FAILED','type':type(error).__name__,'condition':str(error),
              'incurred_oracle_ledger':ledger,'inclusive_wall_seconds':time.monotonic()-started,
              'automatic_retry':False,'scientific_fit_admitted':False})
        raise
    finally:
        rows = [{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file() and not p.name.endswith('.tmp')]
        write(output/'FINAL_CUSTODY.json',{'stage':'fabricated_cpu','completed':(output/'QUALIFICATION.json').exists(),'files':rows})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
