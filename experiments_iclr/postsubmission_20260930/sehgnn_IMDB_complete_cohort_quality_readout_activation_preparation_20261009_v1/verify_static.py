"""Source/structural/default-inactive verification only; no allocation or arrays."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    for path in HERE.glob('*.json'):
        json.loads(path.read_text())
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        path=(HERE.parent/row['path']).resolve(strict=True)
        assert path.is_relative_to(HERE.parent) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    tree=ast.parse((HERE/'ASSEMBLE_AND_RUN.py').read_text())
    calls=[]
    for node in ast.walk(tree):
        assert not isinstance(node,ast.While), 'No watcher/retry loop'
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            calls.append(name)
            assert name not in {'fit','fit_body','train','forward','evaluate','make_model','runtime','load','sleep','Popen','kill','killpg','mount','sudo'}
    assert calls.count('run')==1 and calls.count('observe_absence')==1
    code=(HERE/'ASSEMBLE_AND_RUN.py').read_text()
    assert code.index('evidence=observe_absence(launches)')<code.index("result=read(cell/'RESULT.json')")<code.index('checkpoint=binding(cell/name)')
    assert "CUDA_VISIBLE_DEVICES=''" in code and 'reader.family_owner(' in code and 'reader.entry_owner(' in code and 'reader.custody(custody)' in code
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    body_calls=[node for node in ast.walk(main) if isinstance(node,ast.Call)]
    guards=[node.lineno for node in body_calls if isinstance(node.func,ast.Name) and node.func.id=='guard']
    source_line=next(node.lineno for node in body_calls if isinstance(node.func,ast.Name) and node.func.id=='source_gate')
    run_line=next(node.lineno for node in body_calls if isinstance(node.func,ast.Attribute) and node.func.attr=='run')
    assert len(guards)==2 and min(guards)<source_line and max(guards)<run_line
    cfg=json.loads((HERE/'ACTIVATION.disabled.json').read_text())
    for key in ('enabled','root_activation_source_review_approved','root_readout_execution_approved','root_full18_native5_reference24_and_actual_custody_closure_confirmed'):
        assert cfg[key] is False
    assert cfg['source_seal_sha256'] is cfg['entry_sha256'] is None and cfg['automatic_retry'] is False
    assert cfg['originals']==json.loads((HERE/'PRODUCER_BINDINGS.json').read_text())['originals']
    obs=json.loads((HERE/'METADATA_OBSERVATION.json').read_text())
    assert obs['hostname']=='anogena-2-0' and obs['sole_cuda_uuid']=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    assert obs['shared_reports_results_metrics_or_artifacts_opened'] is False and obs['scientific_outcomes_or_arrays_opened'] is False
    invocation=json.loads((HERE/'INVOCATION.json').read_text())
    assert invocation['ssh_argv_prefix'][-1]=='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
    assert invocation['remote_source'].index("assert socket.gethostname()=='anogena-2-0'")<invocation['remote_source'].index('os.execv(')
    if (HERE/'SEAL.json').exists():
        seal=json.loads((HERE/'SEAL.json').read_text());assert seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json')==seal['manifest_sha256']
        for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
            path=HERE/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    default=subprocess.run([sys.executable,'-B',str(HERE/'ASSEMBLE_AND_RUN.py')],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout)==dict(inactive=True,allocation_queries_or_project_outcomes_read=False,reader_invoked=False) and not default.stderr
    return dict(status='passed',parse_hash_structural_and_disabled_default_only=True,reader_and_analysis_unchanged=True,
        no_scientific_outcomes_checkpoint_arrays_numerical_parity_or_server_execution_in_verification=True,
        one_reader_call_and_one_finite_absence_snapshot=True,root_full_completion_custody_and_execution_approval_required=True)


if __name__=='__main__':
    result=check();target=HERE/'STATIC_VERIFICATION.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
