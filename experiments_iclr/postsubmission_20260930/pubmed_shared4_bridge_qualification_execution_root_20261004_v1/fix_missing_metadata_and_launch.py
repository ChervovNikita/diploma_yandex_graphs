"""Complete a missing metadata dependency before the first numerical child.

The original detached launch/log remain intact. No scientific or engineering
child had been admitted; the immutable source, release and run01 are reused.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
from stage_source_metadata import HERE, PHASE, REPO, REMOTE_PHASE, REMOTE, SOURCE, run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


plan = json.loads((SOURCE / 'PLAN.json').read_text())
release = HERE / 'ROOT_RELEASE_qualification.json'
receipt = plan['input_authority']['feature_equivalence_receipt']
local = PHASE / receipt['path']
assert sha(local) == receipt['sha256']
error_path = HERE / 'owned_monitor01/DETACHED_STDERR.txt'
assert 'FileNotFoundError' in error_path.read_text()
assert receipt['path'] in error_path.read_text()
original_launch = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'assert not Path("/proc/' + str(original_launch['supervisor_PID']) + '").exists()\n'
code += 'assert not (root/"qualification/run01").exists() and not (root/"supervision/qualification/run01").exists()\n'
code += 'error_path=root/"DETACHED_STDERR.txt";assert hashlib.sha256(error_path.read_bytes()).hexdigest()==' + repr(sha(error_path)) + '\n'
code += 'target=phase/' + repr(receipt['path']) + ';raw=base64.b64decode(' + repr(base64.b64encode(local.read_bytes()).decode()) + ',validate=True)\n'
code += 'assert hashlib.sha256(raw).hexdigest()==' + repr(receipt['sha256']) + '\n'
code += 'if target.exists():\n assert not target.is_symlink() and target.read_bytes()==raw\nelse:\n target.parent.mkdir(parents=True,exist_ok=True)\n with target.open("xb") as stream:stream.write(raw)\n'
code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="MISSING_METADATA_STAGED_BEFORE_FIRST_NUMERICAL_CHILD",path=' + repr(receipt['path']) + ',sha256=' + repr(receipt['sha256']) + ',previous_supervisor_absent=True,previous_child_never_admitted=True,original_detached_evidence_preserved=True,numerical_execution=False)))\n'
staged = run('pubmed_bridge_qual_v2_missing_feature_metadata_fix_20261004', code)
save(HERE / 'METADATA_FIX_RECEIPT.json', staged)

# Exercise every standard-library gate before another supervisor invocation.
environment = dict(original_launch['environment'])
profile = plan['execution_profile']
gate_code = 'import sys\nfrom pathlib import Path\nsys.path.insert(0,' + repr(str(REMOTE_PHASE / SOURCE.name)) + ')\nfrom common import gate\ngate(Path(' + repr(str(REMOTE / release.name)) + '),' + repr(sha(release)) + ',"qualification")\nprint("EXACT_STDLIB_GATE_PASS")\n'
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'environment=' + repr(environment) + ';cmd=' + repr([profile['interpreter_path'], '-B', '-c', gate_code]) + '\n'
code += 'result=subprocess.run(cmd,cwd=repo,env=dict(os.environ,**environment),capture_output=True,text=True,timeout=45)\n'
code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="PASS" if result.returncode==0 else "FAILED",exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,numerical_execution=False)))\n'
gate_result = run('pubmed_bridge_qual_v2_full_metadata_gate_20261004', code)
save(HERE / 'PRENUMERICAL_METADATA_GATE.json', gate_result)
assert gate_result['status'] == 'PASS', gate_result

admission = dict(UTC=datetime.now(timezone.utc).isoformat(), status='APPROVED_FIRST_NUMERICAL_CHILD_AFTER_METADATA_FIX',
                 previous_attempt='Exited at metadata gate before output or child creation; retained detached evidence.',
                 immutable_source_manifest_sha256=sha(SOURCE / 'MANIFEST.json'), immutable_release_sha256=sha(release),
                 metadata_fix_sha256=sha(HERE / 'METADATA_FIX_RECEIPT.json'), complete_stdlib_gate_sha256=sha(HERE / 'PRENUMERICAL_METADATA_GATE.json'),
                 numerical_source_model_data_profile_tolerances_changed=False, automatic_retry=False, scientific_fits=0, TEST_access=False)
save(HERE / 'ROOT_METADATA_FIX_ADMISSION.json', admission)
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');root=Path(' + repr(str(REMOTE)) + ');assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'assert not (root/"qualification/run01").exists() and not (root/"supervision/qualification/run01").exists()\n'
raw = (HERE / 'ROOT_METADATA_FIX_ADMISSION.json').read_bytes()
code += 'raw=base64.b64decode(' + repr(base64.b64encode(raw).decode()) + ',validate=True);assert hashlib.sha256(raw).hexdigest()==' + repr(hashlib.sha256(raw).hexdigest()) + '\n'
code += 'with (root/"ROOT_METADATA_FIX_ADMISSION.json").open("xb") as stream:stream.write(raw)\n'
code += 'environment=' + repr(environment) + ';command=' + repr(original_launch['command']) + '\n'
code += 'with (root/"DETACHED_AFTER_METADATA_FIX_STDOUT.txt").open("xb") as out,(root/"DETACHED_AFTER_METADATA_FIX_STDERR.txt").open("xb") as err:\n proc=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
code += 'raw=(Path("/proc")/str(proc.pid)/"stat").read_text();fields=raw[raw.rfind(")")+2:].split()\n'
code += 'result=dict(UTC=datetime.now(timezone.utc).isoformat(),status="ENGINEERING_SUPERVISOR_LAUNCHED_AFTER_METADATA_FIX",supervisor_PID=proc.pid,supervisor_start_ticks=int(fields[19]),command=command,environment=environment,science_fits=0,TEST_access=False,own_engineering_updates_planned=504,original_launch_and_logs_preserved=True)\n'
code += 'with (root/"DETACHED_LAUNCH_AFTER_METADATA_FIX.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\\n")\nprint(json.dumps(result))\n'
result = run('pubmed_bridge_qual_v2_first_numerical_child_launch_20261004', code)
save(HERE / 'DETACHED_LAUNCH_AFTER_METADATA_FIX.json', result)
print(json.dumps({k: result[k] for k in ('UTC', 'status', 'supervisor_PID', 'supervisor_start_ticks')}))
