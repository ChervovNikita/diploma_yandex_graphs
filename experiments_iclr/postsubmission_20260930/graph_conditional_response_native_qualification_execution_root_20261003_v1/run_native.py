"""Run all eight source-defined native CPU families under explicit root release."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PACKET = PHASE / 'graph_conditional_response_native_source_preparation_20261003_v4'


def main():
    authorization_path = HERE / 'ROOT_CPU_ADMISSION.json'
    authorization = json.loads(authorization_path.read_text())
    manifest = PACKET / 'MANIFEST.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == authorization['packet_manifest_sha256']
    for row in json.loads(manifest.read_text())['files']:
        path = PACKET / row['path']
        assert path.resolve().is_relative_to(PACKET.resolve())
        assert path.stat().st_size == row['size']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    assert authorization['explicit_native_cpu_engineering_checks_authorized'] is True
    assert authorization['benchmark_or_predictive_execution'] is False
    assert Path(sys.executable).resolve() == Path(authorization['interpreter_path']).resolve()
    assert hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == authorization['interpreter_sha256']
    sys.path.insert(0, str(PACKET))
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from prepared_native_checks import run_authorized_native_checks
    result = run_authorized_native_checks(str(authorization_path), receipt_file=str(HERE / 'run01/CHECK_RECEIPT.json'))
    result['UTC_finished'] = datetime.now(timezone.utc).isoformat()
    result['root_admission_sha256'] = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    result['executed_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (HERE / 'run01/RESULT.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'status': 'ALL_EIGHT_NATIVE_CPU_FAMILIES_PASSED', 'families': len(result['prepared_native_checks_passed'])}), flush=True)


if __name__ == '__main__':
    main()
