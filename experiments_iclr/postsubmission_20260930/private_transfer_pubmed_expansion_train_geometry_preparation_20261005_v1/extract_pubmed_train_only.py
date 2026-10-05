"""Disabled allocation-only extraction of one authenticated TRAIN member.

No network, feature/heldout/model loader, or import-time data access. A reviewed
root job and external bound are required before this program can execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import tarfile
import time

PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
ARCHIVE = PHASE/'citeseer_heart_official_acquisition_server_20261005_v1/withheld_inputs/HeaRT.tar.gz'
ARCHIVE_SHA = '7b7042476319a353bdb6c50b5f402b89b9006a2fde2d1258b7adcbd1d22629ba'
ARCHIVE_BYTES = 880240878
MEMBER = 'dataset/pubmed/train_pos.txt'
TRAIN_SHA = 'c6de89d86371909f738d620846540168d4b6256fed88dc9d8ab3609cb5357fb4'
TRAIN_BYTES, TRAIN_ROWS, NODES = 409212, 37676, 19717
SOFT_SECONDS, HARD_SECONDS = 600, 900


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', dest='job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = Path(__file__).resolve().parent
    if platform.system() != 'Linux' or socket.gethostname() != 'anogena-2-0':
        raise ValueError('Reviewed allocation host only')
    phase = PHASE.resolve(strict=True)
    job_path = args.job.resolve(strict=True)
    if not source.is_relative_to(phase) or not job_path.is_relative_to(phase):
        raise ValueError('Source/job must remain inside the allocation project phase')
    job = json.loads(job_path.read_text())
    for key in ('source_review_approved', 'TRAIN_member_extraction_authorized',
                'external_900_second_hard_bound_confirmed'):
        if job.get(key) is not True:
            raise ValueError('Separate root review/bounds required: '+key)
    for key in ('network_access', 'VALID_TEST_member_access', 'feature_payload_access',
                'model_access', 'fits_authorized', 'retry'):
        if job.get(key) is not False:
            raise ValueError('TRAIN-only extraction scope differs: '+key)
    if job.get('root_review_reference', '').startswith('PENDING') or not job.get('root_review_reference'):
        raise ValueError('Concrete root review reference required')
    if job.get('extraction_bounds') != {'soft_seconds':SOFT_SECONDS, 'hard_seconds':HARD_SECONDS}:
        raise ValueError('Reviewed prospective bounds differ')
    manifest_path = (source/'SOURCE_MANIFEST.json').resolve(strict=True)
    if not manifest_path.is_relative_to(phase):
        raise ValueError('Source metadata leaves the allocation phase')
    if sha(__file__) != job.get('source_sha256', {}).get(Path(__file__).name):
        raise ValueError('Reviewed extraction source changed')
    if job.get('archive_path') != str(ARCHIVE) or job.get('member') != MEMBER:
        raise ValueError('Only the exact resident archive and Pubmed TRAIN member are allowed')
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(phase) or not output.parent.is_dir() or str(output) != job.get('extraction_output_directory'):
        raise ValueError('Reviewed fresh allocation output required')
    if ARCHIVE.is_symlink() or not ARCHIVE.is_file() or ARCHIVE.stat().st_size != ARCHIVE_BYTES:
        raise ValueError('Resident archive missing or different; no download or fallback')
    if not ARCHIVE.resolve(strict=True).is_relative_to(phase):
        raise ValueError('Archive path leaves the allocation phase')
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Full resident archive authentication failed')
    if time.monotonic()-started > SOFT_SECONDS:
        raise TimeoutError('TRAIN extraction soft bound reached')
    output.mkdir(exist_ok=False)
    try:
        # Tar header traversal is necessary for the exact member lookup. Only
        # this single member's content is extracted or interpreted; no extractall.
        with tarfile.open(ARCHIVE, 'r:gz') as archive:
            matches = [entry for entry in archive if entry.name == MEMBER]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].issym() or matches[0].islnk() or matches[0].size != TRAIN_BYTES:
                raise ValueError('Exact regular TRAIN member identity differs')
            if time.monotonic()-started > SOFT_SECONDS:
                raise TimeoutError('TRAIN extraction soft bound reached')
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError('TRAIN member unreadable')
            with stream, (output/'train_pos.txt').open('xb') as dest:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    dest.write(block)
        target = output/'train_pos.txt'
        if target.stat().st_size != TRAIN_BYTES or sha(target) != TRAIN_SHA:
            raise ValueError('Authenticated TRAIN member bytes differ')
        rows, facts = 0, set()
        with target.open() as stream:
            for line in stream:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    raise ValueError('Native TRAIN format differs')
                u, v = map(int, fields)
                fact = tuple(sorted((u, v)))
                if not 0 <= u < NODES or not 0 <= v < NODES or u == v or fact in facts:
                    raise ValueError('TRAIN population/unique-nonself contract differs')
                facts.add(fact); rows += 1
        if rows != TRAIN_ROWS or time.monotonic()-started > SOFT_SECONDS:
            raise ValueError('TRAIN count or prospective soft bound differs')
        receipt = {'scope':'authenticated_Pubmed_TRAIN_member_only', 'status':'TRAIN_EXTRACTED_NOT_TRAINING_ADMITTED',
            'archive_sha256':ARCHIVE_SHA, 'member':MEMBER, 'train_sha256':TRAIN_SHA,
            'bytes':TRAIN_BYTES, 'rows':rows, 'nodes_declared_from_saved_source':NODES,
            'VALID_TEST_feature_model_access':False, 'network_access':False, 'fits':0,
            'source_manifest_sha256':sha(manifest_path), 'job_sha256':sha(job_path),
            'elapsed_seconds':time.monotonic()-started}
        (output/'TRAIN_EXTRACTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    except Exception as error:
        (output/'FAILURE.json').write_text(json.dumps({'status':'FAILED_NO_ADMISSION',
            'error':type(error).__name__+': '+str(error), 'partial_files_preserved':True,
            'retry':False, 'elapsed_seconds':time.monotonic()-started}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
