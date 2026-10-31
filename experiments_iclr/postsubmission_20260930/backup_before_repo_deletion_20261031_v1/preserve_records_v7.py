"""Preserve code and scientific records as inert data, without loading model states.

The owner requested code/results only. This program never deletes source files,
executes archived files, changes Git, follows links, or launches experiments.
"""
from collections import Counter, defaultdict, OrderedDict
import gzip
import bz2
import getpass
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import tarfile
import time
import zipfile

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
HERE = ROOT / 'experiments_iclr/postsubmission_20260930/backup_before_repo_deletion_20261031_v1'
OUT = HERE / 'records_v7'
DIRECTORY_FDS = OrderedDict()
TEXT = {'.py','.sh','.bash','.json','.jsonl','.ndjson','.csv','.tsv','.txt','.log','.md',
        '.yaml','.yml','.toml','.tex','.bib','.ipynb','.ini','.cfg','.rst','.html','.r','.jl',
        '.sbatch','.lock','.stdout','.stderr','.source','.patch','.diff','.xml','.pyi',
        '.c','.cpp','.h','.hpp','.js','.ts','.css','.sty','.cls','.bst','.out','.err'}
FIGURE = {'.pdf','.png','.jpg','.jpeg','.svg','.eps','.webp'}
STATE = {'.pt','.pth','.ckpt','.pkl','.pickle','.h5','.hdf5','.safetensors'}
ARRAY = {'.npy','.npz','.parquet','.feather'}
RUNTIME = {'.venv','.tools','.cache','.gnnm_runtime','__pycache__','site-packages',
           'site_packages','node_modules','.git','industrial_uv'}
DATA = {'data','raw','processed','provider','provider_cache','working_data','node_bases',
        'datasets','prepared_data','raw_provider_corpus','withheld_inputs','development_inputs'}
TEXT_NAMES = {'LICENSE','Dockerfile','Makefile','.gitignore','STOP','STOP_REQUEST','AUTHORS'}
CHUNK_RAW = 64 * 1024**2
PART = 32 * 1024**2
PRIVATE_LITERALS = []
ARCHIVE_ENDINGS = ('.tar.gz','.tgz','.tar','.zip','.tar.xz','.txz','.tar.bz2','.tbz2')
SECRET_PATTERNS = [
    ('private_key_block', re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----', re.S)),
    ('provider_token', re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,})\b')),
    ('literal_password_assignment', re.compile(rb'(?i)(?:password|passphrase|passwd|access_token|api_key|secret_key)[\"\']?\s*[:=]\s*[\"\']([^\"\'\r\n]{3,200})[\"\']')),
    ('historical_server_password', re.compile(rb'\bOctober[A-Za-z]+20[0-9]{2}[A-Za-z]*\b')),
    ('credential_url', re.compile(rb'https?://[^\s/:@]+:[^\s/@]+@')),
]


def safe_name(value):
    p = PurePosixPath(value)
    return not p.is_absolute() and '..' not in p.parts


def directory_fd(relative):
    """Walk from a verified root FD without following any symlink component."""
    relative = str(relative)
    if relative in DIRECTORY_FDS:
        fd = DIRECTORY_FDS.pop(relative); DIRECTORY_FDS[relative] = fd
        return fd
    if relative == '.':
        fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        assert info.st_dev == 49 and info.st_ino == 9257095286857617644
    else:
        path = Path(relative); assert safe_name(relative)
        parent = directory_fd(path.parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        assert os.fstat(fd).st_dev == 49, 'Nested filesystem is outside repository scope'
    DIRECTORY_FDS[relative] = fd
    while len(DIRECTORY_FDS) > 128:
        name, old = DIRECTORY_FDS.popitem(last=False)
        if name == '.': DIRECTORY_FDS[name] = old
        else: os.close(old)
    return fd


def open_record(path):
    relative = path.relative_to(ROOT)
    fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd(relative.parent))
    info = os.fstat(fd)
    assert stat.S_ISREG(info.st_mode) and info.st_dev == 49
    return os.fdopen(fd,'rb'), info


def choice(value, category='', size=0):
    p = PurePosixPath(value); parts = set(p.parts); low = value.lower(); name = p.name.lower()
    suffix = p.suffix.lower()
    if parts & RUNTIME or category == 'reproducible_runtime_or_vendor_git':
        return False, 'installed_runtime_or_vendor_git'
    if 'industrial_runtime_image_root_v1/runtime_image_v1/' in low or '/envs/' in low or '/cache/pip/' in low:
        return False, 'installed_runtime_or_cache'
    if name.endswith('.b64') or suffix in {'.pyc','.so','.whl','.a','.o','.enc','.rkyv'}:
        return False, 'compiled_runtime_or_transport_duplicate'
    if name == 'cycle_draws.jsonl.gz':
        return False, 'raw_sampling_replay_trace_excluded_by_scope'
    if name in {'.env','.netrc','credentials','id_rsa','id_ed25519'} or name.endswith(('.pem','.key')):
        return False, 'private_credentials_not_published'
    if suffix in TEXT or p.name in TEXT_NAMES or suffix in FIGURE:
        if name == 'data.json' and ({'wiki_cs','wikics','official'} & parts):
            return False, 'downloaded_public_dataset'
        if suffix in {'.csv','.tsv','.txt'} and parts & DATA and not re.search(r'manifest|metric|score|result|curve|history|audit|readme|license|requirement|split|indice|mask', name):
            return False, 'downloaded_raw_dataset_table'
        return True, 'source_or_scientific_record'
    if name.endswith(('.jsonl.gz','.json.gz','.csv.gz','.txt.gz','.tsv.gz')):
        if parts & {'raw','provider','provider_cache'} or name == 'mol.csv.gz':
            return False, 'downloaded_public_dataset'
        return True, 'compressed_scientific_record'
    if suffix in STATE | ARRAY:
        if parts & {'checkpoints','checkpoint','model_checkpoints'} or re.search(r'checkpoint|ckpt|recovery|optimizer|(?:^|_)state(?:\.|_)', name):
            return False, 'checkpoint_or_optimizer_state_excluded_by_owner'
        if re.search(r'logit|predict|embedding|posterior|metric|score|(?:^|_)mask(?:\.|_)|(?:^|_)indices(?:\.|_)', name):
            return True, 'prediction_or_protocol_result_array'
        if parts & DATA or re.search(r'graph_training|^graph\.|^features?\.|^feats?\.|^edges?\.|^labels?\.|^download_', name):
            return False, 'downloaded_or_derived_dataset'
        if suffix in STATE:
            if re.search(r'^split|^train_idx|^val_idx|^test_idx', name):
                return True, 'split_protocol_array'
            return False, 'training_state_or_derived_tensor_payload_excluded'
        if re.search(r'split|indice|mask', name):
            return True, 'split_protocol_array'
        return True, 'scientific_array_export'
    if name.endswith(ARCHIVE_ENDINGS):
        return None, 'archive_members_selected_individually'
    if suffix in {'.xz','.bz2'}:
        return True, 'compressed_historical_record'
    return False, 'nonrecord_binary_or_dataset_auxiliary'


def clean_bytes(raw, name):
    """Redact plausible literal credentials; return only rule labels, never values."""
    compressed = name.lower().endswith(('.jsonl.gz','.json.gz','.csv.gz','.txt.gz','.tsv.gz'))
    codec = 'gzip' if compressed else ('xz' if name.endswith('.xz') else ('bz2' if name.endswith('.bz2') else None))
    scan = gzip.decompress(raw) if codec == 'gzip' else (lzma.decompress(raw) if codec == 'xz' else (bz2.decompress(raw) if codec == 'bz2' else raw))
    if b'\x00' in scan[:8192] and not compressed:
        return raw, []
    reasons = []
    for secret in PRIVATE_LITERALS:
        if secret not in scan:
            continue
        if len(secret) >= 6:
            pattern = re.compile(re.escape(secret))
        else:
            pattern = re.compile(rb'(?<![A-Za-z0-9_])' + re.escape(secret) + rb'(?![A-Za-z0-9_])')
        if pattern.search(scan):
            reasons.append('known_private_literal')
            scan = pattern.sub(b'[REDACTED_CREDENTIAL]', scan)
    for label, pattern in SECRET_PATTERNS:
        def replacement(match):
            if label == 'literal_password_assignment':
                value = match.group(1)
                # Expressions/placeholders do not contain a saved credential.
                if any(v in value.lower() for v in (b'redact', b'placeholder', b'getenv', b'environ', b'password', b'passphrase', b'example', b'none', b'null', b'false', b'true', b'path', b'input', b'getpass', b'argv', b'not_saved', b'not-stored', b'not stored')) or value.startswith((b'$',b'<',b'{')):
                    return match.group(0)
                reasons.append(label)
                start, end = match.span(1)
                return match.group(0)[:start-match.start()] + b'[REDACTED_CREDENTIAL]' + match.group(0)[end-match.start():]
            reasons.append(label)
            return b'[REDACTED_CREDENTIAL]'
        scan = pattern.sub(replacement, scan)
    if not reasons:
        return raw, []
    if codec == 'gzip': scan = gzip.compress(scan, compresslevel=6, mtime=0)
    elif codec == 'xz': scan = lzma.compress(scan)
    elif codec == 'bz2': scan = bz2.compress(scan)
    return scan, sorted(set(reasons))


class Store:
    def __init__(self):
        self.parts = {}; self.objects = {}; self.chunks = []; self.tar = None
        self.raw_size = 0; self.total_unique_bytes = 0
    def close_chunk(self):
        if self.tar is None:
            return
        self.tar.close()
        path = self.chunk_path
        self.chunks.append({'path': str(path.relative_to(OUT)), 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        assert path.stat().st_size < 90 * 1024**2
        self.tar = None
    def add(self, raw):
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self.objects:
            return digest
        pieces = []
        for offset in range(0, max(len(raw), 1), PART):
            data = raw[offset:offset+PART]; sha = hashlib.sha256(data).hexdigest()
            if sha not in self.parts:
                if self.tar is None or self.raw_size + len(data) > CHUNK_RAW:
                    self.close_chunk()
                    self.chunk_path = OUT / 'chunks' / ('records-%04d.tar.gz' % len(self.chunks))
                    self.tar = tarfile.open(str(self.chunk_path), 'w:gz', compresslevel=6)
                    self.raw_size = 0
                item = tarfile.TarInfo(sha); item.size = len(data); item.mode = 0o644; item.mtime = 0
                self.tar.addfile(item, io.BytesIO(data)); self.raw_size += len(data)
                self.parts[sha] = {'chunk': str(self.chunk_path.relative_to(OUT)), 'bytes': len(data)}
            pieces.append(sha)
        self.objects[digest] = {'bytes': len(raw), 'parts': pieces}
        self.total_unique_bytes += len(raw)
        return digest


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert ROOT.resolve() == ROOT and ROOT.stat().st_dev == 49 and ROOT.stat().st_ino == 9257095286857617644
    for number in range(3):
        value = getpass.getpass('Private literal for publication scan %d/3 (not saved): ' % (number+1))
        if value: PRIVATE_LITERALS.append(value.encode())
    assert not OUT.exists(), 'A prior preparation exists; inspect before retrying.'
    (OUT / 'chunks').mkdir(parents=True)
    store = Store(); counts = Counter(); sizes = Counter(); redactions = []; variations = []; archives = []
    started = time.time(); seen = {}; processed = 0
    # Reuse only data parts verified from the interrupted preparation. No file
    # from that archive is executed. This avoids repeating costly compression.
    recovered = []
    # The most recent chunk may have been open at controlled interruption; omit
    # it and regenerate its data from originals. Earlier chunks are closed.
    for prior in sorted((HERE / 'records_v6/chunks').glob('*.tar.gz'))[:-1]:
        target = OUT / 'chunks' / prior.name
        with tarfile.open(str(prior),'r|gz') as packed:
            for member in packed:
                assert member.isfile() and re.fullmatch('[0-9a-f]{64}',member.name)
                data = packed.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == member.name
                store.parts[member.name] = {'chunk':'chunks/'+prior.name,'bytes':len(data)}
        os.link(prior,target)
        receipt = {'path':'chunks/'+prior.name,'bytes':prior.stat().st_size,'sha256':hashlib.sha256(prior.read_bytes()).hexdigest()}
        store.chunks.append(receipt); recovered.append(receipt)
    # Recover complete JSON rows from the interrupted gzip stream. Its missing
    # footer is expected; every reused content part was independently verified.
    previous_checkpoint = json.loads((HERE / 'records_v6/CHECKPOINT.json').read_text())
    for sha, value in previous_checkpoint['objects'].items():
        if all(part in store.parts for part in value['parts']):
            store.objects[sha] = value
    prior_redactions = {row['path']:row['rules'] for row in previous_checkpoint['redactions']}
    prior_archives = {row['path']:row for row in previous_checkpoint['archives']}
    variations.extend(previous_checkpoint['variations'])
    groups = defaultdict(list)
    prefix_rows = 0
    with gzip.open(HERE / 'records_v6/COVERAGE.jsonl.gz', 'rt') as prior_coverage:
        while True:
            try:
                line = prior_coverage.readline()
            except EOFError:
                break
            if not line or not line.endswith('\n'):
                break
            row = json.loads(line)
            groups[row['path'].split('::', 1)[0]].append(row)
            prefix_rows += 1
            sha = row.get('preserved_object')
            if row['status'].startswith('preserved_'):
                if sha not in store.objects and sha in store.parts:
                    store.objects[sha] = {'bytes': store.parts[sha]['bytes'], 'parts': [sha]}
                if sha in store.objects:
                    if row['status'] == 'preserved_exact':
                        assert row['original_sha256'] == sha and row['bytes'] == store.objects[sha]['bytes']
                        flags = []
                    else:
                        flags = prior_redactions.get(row['path'], ['verified_redaction_from_prior_scan'])
                    seen[row['original_sha256']] = (sha, flags)
    store.total_unique_bytes = sum(row['bytes'] for row in store.objects.values())
    resumed_groups = {}
    for source_path, rows in groups.items():
        # An archive source is complete only after its top-level container row.
        complete_source = any(row['path'] == source_path for row in rows)
        recoverable = all(row['status'] == 'excluded' or
                          (row['status'].startswith('preserved_') and row.get('preserved_object') in store.objects)
                          for row in rows)
        if complete_source and recoverable:
            resumed_groups[source_path] = rows
    (OUT/'RECOVERED_PREFIX.json').write_text(json.dumps({'reused_chunks':recovered,
        'source_run_status':'v6_controlled_stop_after_equivalent_prefilter_benchmark','all_part_hashes_verified':True,
        'complete_prefix_rows':prefix_rows,'resumed_source_files':len(resumed_groups),
        'resumed_path_rows':sum(map(len,resumed_groups.values())),
        'multipart_maps_recovered_from_checkpoint':True,'only_unmapped_records_reread':True},indent=2)+'\n')
    print(json.dumps({'reused_verified_chunks':len(recovered),'reused_verified_parts':len(store.parts),
        'resumed_source_files':len(resumed_groups),'resumed_path_rows':sum(map(len,resumed_groups.values())),
        'elapsed_seconds':round(time.time()-started)}),flush=True)
    coverage = gzip.open(OUT / 'COVERAGE.jsonl.gz', 'wt', encoding='utf-8')

    def checkpoint():
        """Durable object maps prevent losing multipart boundaries on interruption."""
        store.close_chunk()
        coverage.flush()
        value = {'objects':store.objects,'parts':store.parts,'chunks':store.chunks,
                 'counts':dict(counts),'logical_bytes':dict(sizes),'archives':archives,
                 'redactions':redactions,'variations':variations,
                 'completed_source_files':processed,'coverage_gzip_footer_pending':True}
        temporary = OUT / 'CHECKPOINT.next.json'
        with temporary.open('w') as stream:
            json.dump(value,stream,separators=(',',':')); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(OUT / 'CHECKPOINT.json')

    for source_path, rows in resumed_groups.items():
        for row in rows:
            coverage.write(json.dumps(row,separators=(',',':'))+'\n')
            key = row['status'] if row['status'].startswith('preserved_') else row['reason']
            counts[key] += 1; sizes[key] += row['bytes']
            if row['status'] == 'preserved_redacted':
                redactions.append({'path':row['path'],'rules':seen[row['original_sha256']][1],
                                   'original_sha256':row['original_sha256']})
        if any(row['reason'] == 'archive_container_replaced_by_selected_records' for row in rows):
            archives.append(prior_archives.get(source_path, {'path':source_path,'recovered_complete_archive_coverage':True,
                'selected_record_count':sum(row['status'].startswith('preserved_') for row in rows),
                'archived_files_executed':False}))
    checkpoint()

    def record(path, raw, reason, extra=None):
        original_sha = hashlib.sha256(raw).hexdigest()
        if original_sha not in seen:
            clean, flags = clean_bytes(raw, path)
            kept_sha = store.add(clean)
            seen[original_sha] = (kept_sha, flags)
        kept_sha, flags = seen[original_sha]
        row = {'path': path, 'bytes': len(raw), 'status': 'preserved_redacted' if flags else 'preserved_exact',
               'reason': reason, 'original_sha256': original_sha, 'preserved_object': kept_sha}
        if extra: row.update(extra)
        if flags: redactions.append({'path': path, 'rules': flags, 'original_sha256': original_sha})
        coverage.write(json.dumps(row, separators=(',',':'))+'\n')
        counts[row['status']] += 1; sizes[row['status']] += len(raw)

    def omit(path, size, reason, extra=None):
        row = {'path': path, 'bytes': size, 'status': 'excluded', 'reason': reason}
        if extra: row.update(extra)
        coverage.write(json.dumps(row,separators=(',',':'))+'\n')
        counts[reason] += 1; sizes[reason] += size

    def archive(path):
        relative = str(path.relative_to(ROOT)); count = 0; selected = 0; manifest = None; object_rows = defaultdict(list)
        def nested(raw, prefix, filename):
            if filename.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for info in z.infolist():
                        if not info.is_dir():
                            with z.open(info) as f: member(info.filename, info.file_size, f, prefix)
            else:
                with tarfile.open(fileobj=io.BytesIO(raw), mode='r|*') as tar:
                    for info in tar:
                        if info.isfile(): member(info.name, info.size, tar.extractfile(info), prefix)
        def member(name, size, stream, prefix=None):
            nonlocal count, selected, manifest
            assert safe_name(name), 'Unsafe archive name'
            count += 1
            label = (prefix or relative) + '::' + name
            if name in {'EVIDENCE_MANIFEST.json','COLD_FILE_MANIFEST.json'}:
                raw = stream.read(); manifest = json.loads(raw)
                for row in manifest['files']: object_rows[row['sha256']].append(row)
                record(label, raw, 'historical_archive_mapping'); selected += 1; return
            if name.startswith('objects/') and manifest is not None:
                digest = name.split('/',1)[1]
                mapped = object_rows[digest]
                wanted = []
                for row in mapped:
                    yes, reason = choice(row['path'], size=row['bytes'])
                    original = relative + '::restored/' + row['path']
                    if yes or yes is None: wanted.append((row, original, reason, yes))
                    else: omit(original, row['bytes'], reason, {'archive_object': name})
                if wanted:
                    raw = stream.read(); assert hashlib.sha256(raw).hexdigest() == digest
                    for row, original, reason, yes in wanted:
                        if yes:
                            record(original, raw, reason, {'archive_object': name, 'original_mode': row.get('mode')}); selected += 1
                        else:
                            nested(raw, original, row['path'])
                            omit(original, row['bytes'], 'nested_archive_replaced_by_selected_records', {'archive_object': name})
                return
            context = name
            if DATA & set(PurePosixPath(relative).parts): context = 'data/' + name
            yes, reason = choice(context, size=size)
            if yes:
                record(label, stream.read(), reason); selected += 1
            elif yes is None:
                nested(stream.read(), label, name)
                omit(label, size, 'nested_archive_replaced_by_selected_records')
            else:
                omit(label, size, reason)
        archive_file, before = open_record(path)
        if path.name.endswith('.zip'):
            with archive_file, zipfile.ZipFile(archive_file) as z:
                for info in z.infolist():
                    if info.is_dir(): continue
                    with z.open(info) as f: member(info.filename, info.file_size, f)
        else:
            with archive_file, tarfile.open(fileobj=archive_file, mode='r|*') as tar:
                for info in tar:
                    if not info.isfile(): continue
                    member(info.name, info.size, tar.extractfile(info))
        archives.append({'path': relative, 'member_count': count, 'selected_record_count': selected,
                         'archived_files_executed': False})

    for line in (HERE / 'REFINED_INVENTORY.jsonl').open():
        row = json.loads(line); p = ROOT / row['path']
        if row['path'] in resumed_groups:
            processed += 1
            continue
        yes, reason = choice(row['path'], row['preservation_category'], row['bytes'])
        if yes:
            try:
                source, before = open_record(p)
            except FileNotFoundError:
                if p.name == 'OWN_WORKER.lock' or p.suffix == '.tmp':
                    omit(row['path'],row['bytes'],'ephemeral_lock_removed_before_preservation'); continue
                if '/shared_disk_and_host_ledgers/reservations/' in row['path'] and p.suffix == '.json':
                    omit(row['path'],row['bytes'],'ephemeral_scheduler_reservation_removed_on_shutdown'); continue
                raise
            with source:
                raw = source.read(); after = os.fstat(source.fileno())
            assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'Record changed during read'
            if len(raw) != row['bytes'] or before.st_mtime_ns != row['mtime_ns']:
                variations.append({'path': row['path'], 'inventory_bytes': row['bytes'], 'current_bytes': len(raw)})
            record(row['path'], raw, reason)
        elif yes is None:
            archive(p)
            omit(row['path'], row['bytes'], 'archive_container_replaced_by_selected_records')
            checkpoint()
        else:
            omit(row['path'], row['bytes'], reason)
        processed += 1
        if processed % 25000 == 0:
            checkpoint()
        if processed % 10000 == 0:
            print(json.dumps({'scanned_files': processed, 'preserved_paths': counts['preserved_exact']+counts['preserved_redacted'],
                              'unique_objects': len(store.objects), 'closed_chunk_bytes': sum(c['bytes'] for c in store.chunks),
                              'elapsed_seconds': round(time.time()-started)}), flush=True)
    coverage.close(); store.close_chunk()
    (OUT / 'OBJECTS.json').write_text(json.dumps({'objects': store.objects, 'parts': store.parts}, separators=(',',':'))+'\n')
    (OUT / 'REDACTIONS.json').write_text(json.dumps(redactions, indent=2)+'\n')
    (OUT / 'ARCHIVES.json').write_text(json.dumps(archives, indent=2)+'\n')
    (OUT / 'VARIATIONS.json').write_text(json.dumps(variations, indent=2)+'\n')
    summary = {'status': 'PREPARED_NOT_YET_PUBLISHED', 'root': str(ROOT), 'counts': dict(counts), 'logical_bytes': dict(sizes),
               'unique_preserved_object_bytes': store.total_unique_bytes, 'unique_objects': len(store.objects),
               'chunks': store.chunks, 'compressed_payload_bytes': sum(c['bytes'] for c in store.chunks),
               'redacted_path_count': len(redactions), 'archive_count': len(archives), 'files_changed_since_inventory': variations,
               'jobs_reported_stopped_by_user': True, 'historical_running_states_preserved_without_reclassification': True,
               'source_records_recalculated': False, 'checkpoints_excluded_by_owner': True, 'elapsed_seconds': time.time()-started,
               'deletion_performed': False, 'git_objects_created': False, 'source_files_executed': False}
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','counts','unique_objects','unique_preserved_object_bytes','compressed_payload_bytes','redacted_path_count','archive_count','elapsed_seconds')}),flush=True)


if __name__ == '__main__':
    main()
