"""Inactive root-approved one continuation: closed original18 origins then centered3."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--release',type=Path);parser.add_argument('--published-commit')
    args=parser.parse_args()
    if not args.execute:print(json.dumps(dict(inactive=True,no_score_array_model_or_process_action=True)));return
    A=Path(__file__).resolve().parent;P=A.parent;R=P.parents[1]
    assert A.name=='geometry_only_core_centered3_activation_root_20261009_v1' and socket.gethostname()=='anogena-2-0'
    assert args.release and args.published_commit
    cfg=json.loads(args.release.read_text())
    assert cfg['enabled'] is True and cfg['release_owner']=='root' and cfg['action']=='original18_origins_then_centered3'
    assert cfg['continuation_source_sha256']==sha(__file__)
    assert cfg['builder_source_sha256']==sha(A/'build_origins.py')
    assert cfg['supervisor_sha256']==sha(A/'SUPERVISOR.py')
    assert cfg['centered_release_template_sha256']==sha(A/'CENTERED_RELEASE_TEMPLATE_DISABLED.json')
    assert str(A)==cfg['activation_directory']
    assert not (A/'LAUNCH.json').exists() and not (A/'OWNER.log').exists() and not (A/'RELEASE.json').exists()
    assert not Path(cfg['centered_output_directory']).exists()
    head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip();assert head==args.published_commit
    subprocess.run(['git','-C',str(R),'merge-base','--is-ancestor','2bfcdf5933e9ad376bda3a7f42a009b1a6cb0d1f',head],check=True)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    spec=importlib.util.spec_from_file_location('_closed_original18_origins',A/'build_origins.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    if (A/'ORIGINS.json').exists():
        builder.closure(cfg)
        approved=cfg['prebuilt_origins']
        assert str((A/'ORIGINS.json').resolve())==approved['path'] and sha(A/'ORIGINS.json')==approved['sha256']
        receipt=json.loads((A/'ORIGINS.json').read_text())
        assert receipt['builder_source_sha256']==cfg['builder_source_sha256']
        assert receipt['original_custody_sha256']==cfg['original18_custody']['sha256']
        assert receipt['original_complete']['sha256']==cfg['original_complete']['sha256']
        assert receipt['all_original_costs_charged'] is True and receipt['comparative_outcomes_unopened'] is True and len(receipt['records'])==18
        origins=dict(path=str(A/'ORIGINS.json'),sha256=sha(A/'ORIGINS.json'))
    else:
        origins=builder.build(cfg,A/'ORIGINS.json')
    release=json.loads((A/'CENTERED_RELEASE_TEMPLATE_DISABLED.json').read_text())
    assert release['enabled'] is False and release['execution_source_commit']==cfg['centered_source_anchor']
    assert release['source_seal_sha256']==cfg['centered_source_seal_sha256']
    assert release['engineering_qualification']==cfg['centered_qualification'] and release['revised_scientific_scope']==cfg['root_scope']
    release.update(enabled=True,original18_origins={key:origins[key] for key in ('path','sha256')},
                   original18_custody=cfg['original18_custody'],origin_builder_source_sha256=cfg['builder_source_sha256'])
    with (A/'RELEASE.json').open('x') as stream:json.dump(release,stream,indent=2,sort_keys=True);stream.write('\n')
    with (A/'OWNER.log').open('xb') as log:
        child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(A/'SUPERVISOR.py')],cwd=R,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    raw=Path('/proc',str(child.pid),'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    parent=dict(pid=child.pid,start_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]),state=fields[0],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    assert parent['group']==parent['session']==child.pid
    launch=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=parent,source_commit=head,
        scientific_source_anchor=release['execution_source_commit'],supervisor_sha256=cfg['supervisor_sha256'],release_sha256=sha(A/'RELEASE.json'),
        origin_receipt_sha256=origins['sha256'],continuation_release_sha256=sha(args.release),centered_shared_fits=3,new_independent_fits=0,
        exact_independent_bodies_reused=12,required_family_records_before_comparison=21,normal_host=True,automatic_retry=False,
        comparative_outcomes_opened=False,TEST_truth_accessed=False)
    with (A/'LAUNCH.json').open('x') as stream:json.dump(launch,stream,indent=2);stream.write('\n')
    print(json.dumps(launch))


if __name__=='__main__':main()
