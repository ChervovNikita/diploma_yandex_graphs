"""Admit two reviewed engineering cases serially with a larger memory limit."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from prepare_molecular_packets_allocation import TRANSFER, invoke
from launch_coauthor_resource_small_queue import REMOTE, PACKAGE

DEST = 'coauthor_resource_larger_reservation_queue_20261012_v1'
LABEL = 'coauthor_resource_larger_reservation_queue_root_20261012_v1'


def main():
    out=PACKAGE/'preparation'/LABEL;out.mkdir(exist_ok=False)
    review=json.loads((PACKAGE/'preparation/coauthor_capacity_deferral_review_root_20261012_v1/ROOT_CAPACITY_REVIEW.json').read_text())
    assert review['single_child_at_a_time'] and review['old_failures_preserved'] and len(review['case_ids'])==2
    raw=(PACKAGE/'qa/coauthor_resource_small_queue.py').read_bytes()
    config={'helper':review['next_helper'],'manifest':review['next_manifest'],'case_ids':review['case_ids'],
            'output':DEST+'/execution_v1',
            'runtime':'massive_polyformer_coauthor_practical_packet_20261012_v1/sources/training_pipeline_v2_20261012'}
    payload={'destination':DEST,'config':config,'queue':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
              'data':base64.b64encode(raw).decode()}}
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    result=invoke(TRANSFER,[DEST,'0',encoded],out,'TRANSFER',60);assert result['exit_code']==0
    marker="with (stage/'owner.log').open('xb') as log:"
    assert REMOTE.count(marker)==1
    source=REMOTE.replace(marker,"free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())\nassert free>=65536,'Current free memory cannot admit the reviewed larger engineering reservation'\n"+marker)
    result=invoke(source,[DEST,hashlib.sha256(encoded.encode()).hexdigest()],out,'DETACHED_LAUNCH',60)
    assert result['exit_code']==0, result['stdout']
    (out/'ROOT_LAUNCH_SCOPE.json').write_text(json.dumps({'case_ids':review['case_ids'],
        'maximum_fraction_of_total_current_child_only':.75,'minimum_free_MiB_at_launch':65536,
        'scientific_fits':0,'GO_created':False,'old_32GiB_failures_preserved':True,
        'hypothesis_or_model_or_optimizer_source_changed':False},indent=2)+'\n')
    print(result['stdout'].strip())


if __name__=='__main__':main()
