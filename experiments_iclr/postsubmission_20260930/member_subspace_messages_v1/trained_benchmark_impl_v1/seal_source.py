"""Seal the new caller packet using stdlib only; no execution admission."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path.relative_to(PHASE)),sha256=sha(path))


def write(name,value):
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    spec=importlib.util.spec_from_file_location('_trained_benchmark_seal_checks',HERE/'verify_stdlib.py')
    check=importlib.util.module_from_spec(spec);sys.modules[spec.name]=check;spec.loader.exec_module(check)
    result=check.verify()
    write('STDLIB_CHECK_v1.json',result)
    scientific=PHASE/'shared_message_predictive_v3'
    policy=copy.deepcopy(json.loads((scientific/'RUNTIME_IMPORT_POLICY_v1.json').read_text()))
    own={str(path.relative_to(PHASE)):sha(path) for path in sorted(HERE.glob('*.py'))}
    policy['importable_sha256'].update(own)
    policy.update(caller_policy_schema='gnnm-trained-inference-augmented-caller-policy-v1',caller_sources=own,
                  inherited_scientific_manifest=ref(scientific/'SOURCE_MANIFEST_v1.json'))
    write('CALLER_IMPORT_POLICY_v1.json',policy)
    protocol=PHASE/'trained_shared_message_benchmark_v1'
    refs=dict(scientific_manifest=ref(scientific/'SOURCE_MANIFEST_v1.json'),scientific_bindings=ref(scientific/'SOURCE_BINDINGS.json'),
              scientific_import_policy=ref(scientific/'RUNTIME_IMPORT_POLICY_v1.json'),unchanged_model_manifest=ref(PHASE/'shared_message_quality_v3/SOURCE_MANIFEST_v1.json'),
              protocol=ref(protocol/'PROTOCOL.md'),protocol_freeze=ref(protocol/'PROTOCOL_FREEZE_before_quality_v1.json'))
    bindings=dict(json.loads((scientific/'SOURCE_BINDINGS.json').read_text())['sha256'])
    bindings.update({r['path']:r['sha256'] for r in refs.values()})
    files=sorted(path for path in HERE.iterdir() if path.is_file() and path.name not in {'SOURCE_BINDINGS.json','SOURCE_MANIFEST_v1.json','SHA256SUMS'})
    bindings.update({str(path.relative_to(PHASE)):sha(path) for path in files})
    write('SOURCE_BINDINGS.json',dict(schema='gnnm-trained-inference-separate-source-bindings-v1',sha256=bindings,execution_authorized=False))
    files.append(HERE/'SOURCE_BINDINGS.json')
    manifest=dict(schema='gnnm-trained-inference-source-manifest-v1',execution_authorized=False,
                  inherited_and_protocol_refs=refs,source_bindings_sha256=sha(HERE/'SOURCE_BINDINGS.json'),
                  files=[dict(relative_path=path.name,bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(files)],
                  source_only_stdlib_checks=result['check_count'],data_Torch_GPU_targets_checkpoints_predictions_or_quality_outcomes_accessed=False)
    write('SOURCE_MANIFEST_v1.json',manifest)
    with (HERE/'SHA256SUMS').open('x') as stream:
        for path in sorted(files+[HERE/'SOURCE_MANIFEST_v1.json']):stream.write(sha(path)+'  '+path.name+'\n')
    print(json.dumps(dict(source_manifest_sha256=sha(HERE/'SOURCE_MANIFEST_v1.json'),source_bindings_sha256=sha(HERE/'SOURCE_BINDINGS.json'),
                          source_only_checks=result['check_count'],execution_authorized=False),sort_keys=True))


if __name__=='__main__':main()
