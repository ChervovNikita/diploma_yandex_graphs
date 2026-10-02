"""Verify deployed bytes and diagnose the separate ancestry-source checker."""
import json
from pathlib import Path
import runpy
import sys
import traceback

BASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
PACKET = BASE / 'industrial_native_pilot_execution_preparation_v2'
sys.path.insert(0, str(PACKET))
from runtime_fingerprint import verify_source_packet

if __name__ == '__main__':
    own = verify_source_packet(PACKET)
    print(json.dumps({'deployed_packet_verified': own, 'scientific_execution': False}), flush=True)
    try:
        runpy.run_path(str(PACKET / 'check_sources.py'), run_name='__main__')
    except Exception as error:
        print(json.dumps({'full_ancestry_source_check_passed': False,
            'failure_type': type(error).__name__, 'failure': str(error),
            'traceback': traceback.format_exc(), 'scientific_execution': False}), flush=True)
        raise SystemExit(1)
