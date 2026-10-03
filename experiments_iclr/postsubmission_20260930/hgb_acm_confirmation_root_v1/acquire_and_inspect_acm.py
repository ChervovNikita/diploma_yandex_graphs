"""Acquire official ACM and inspect topology, features and source development labels."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import traceback
import urllib.request
import zipfile

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
ROOT = REPO / 'experiments_iclr/postsubmission_20260930/hgb_acm_confirmation_root_v1'
FILE_ID = '1xbJ4QE9pcDJOcALv7dYhHDCPITX2Iddz'
URL = 'https://drive.usercontent.google.com/download?id=' + FILE_ID + '&export=download&authuser=0&confirm=t'


def main():
    assert Path.cwd().resolve() == REPO
    assert subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip() == str(REPO)
    assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    locator = json.loads((ROOT / 'OFFICIAL_ARCHIVE_LOCATOR.json').read_text())
    assert locator['filename'] == 'ACM.zip' and locator['file_id'] == FILE_ID
    run = ROOT / 'acquisition01'
    run.mkdir(exist_ok=False)
    row = dict(UTC=datetime.now(timezone.utc).isoformat(), source_URL=URL, file_id=FILE_ID,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               locator_sha256=hashlib.sha256((ROOT / 'OFFICIAL_ARCHIVE_LOCATOR.json').read_bytes()).hexdigest(),
               heldout_label_payload_opened=False, models_executed=False, split_adopted=False)
    partial = run / 'ACM.zip.partial'
    try:
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30) as response:
            row.update(HTTP_status=response.status, final_URL=response.url, content_type=response.headers.get('Content-Type'))
            with partial.open('xb') as stream:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > 250_000_000:
                        raise ValueError('Unexpected archive size; retain partial and inspect source release')
                    stream.write(block)
                    digest.update(block)
        assert zipfile.is_zipfile(partial), 'Official download did not return a ZIP'
        archive = run / 'ACM.zip'
        partial.rename(archive)
        row.update(archive_path=str(archive), archive_bytes=total, archive_sha256=digest.hexdigest())
        hashes = {}

        def lines(z, name):
            digest = hashlib.sha256()
            with z.open(name) as stream:
                for raw in stream:
                    digest.update(raw)
                    yield raw.decode().rstrip('\r\n').split('\t')
            hashes[name] = digest.hexdigest()

        with zipfile.ZipFile(archive) as z:
            members = []
            for member in z.infolist():
                path = PurePosixPath(member.filename)
                assert not path.is_absolute() and '..' not in path.parts
                members.append(dict(path=member.filename, uncompressed_bytes=member.file_size,
                                    compressed_bytes=member.compress_size, CRC32=member.CRC))
            row['archive_members_metadata_only'] = members
            nodes = {}
            counts = Counter()
            widths = defaultdict(Counter)
            missing = Counter()
            identifiers = defaultdict(list)
            for fields in lines(z, 'ACM/node.dat'):
                assert len(fields) in (3, 4)
                nid, nt = int(fields[0]), int(fields[2])
                assert nid not in nodes
                nodes[nid] = nt
                counts[nt] += 1
                identifiers[nt].append(nid)
                if len(fields) == 3 or not fields[3]:
                    missing[nt] += 1
                else:
                    values = fields[3].split(',')
                    widths[nt][len(values)] += 1
                    assert all(math.isfinite(float(v)) for v in values)
            assert set(nodes) == set(range(len(nodes)))
            types = {}
            for nt in sorted(counts):
                ids = sorted(identifiers[nt])
                assert ids == list(range(min(ids), max(ids) + 1)) and len(widths[nt]) <= 1
                types[nt] = dict(nodes=counts[nt], global_id_offset=min(ids),
                                 feature_width_counts=dict(widths[nt]), absent_feature_rows=missing[nt])
            pairs = defaultdict(set)
            edges = Counter()
            weights = defaultdict(Counter)
            endpoints = {}
            self_edges = Counter()
            for fields in lines(z, 'ACM/link.dat'):
                assert len(fields) == 4
                src, dst, rel = map(int, fields[:3])
                weight = float(fields[3])
                assert src in nodes and dst in nodes and math.isfinite(weight)
                pair = (nodes[src], nodes[dst])
                assert rel not in endpoints or endpoints[rel] == pair
                endpoints[rel] = pair
                pairs[rel].add((src, dst))
                edges[rel] += 1
                weights[rel][weight] += 1
                self_edges[rel] += src == dst
            pool = {}
            classes = Counter()
            label_types = Counter()
            for fields in lines(z, 'ACM/label.dat'):
                assert len(fields) == 4
                nid, nt = int(fields[0]), int(fields[2])
                labels = fields[3].split(',')
                assert nid in nodes and nodes[nid] == nt and len(labels) == 1 and nid not in pool
                label = int(labels[0])
                assert label >= 0
                pool[nid] = label
                classes[label] += 1
                label_types[nt] += 1
            assert len(label_types) == 1 and sorted(classes) == list(range(len(classes)))
            target_type = next(iter(label_types))
            relations = {rel: dict(source_type=endpoints[rel][0], target_type=endpoints[rel][1],
                                   raw_records=edges[rel], unique_directed_pairs=len(pairs[rel]),
                                   duplicate_records=edges[rel] - len(pairs[rel]),
                                   self_records=self_edges[rel], raw_weight_counts=dict(weights[rel]))
                         for rel in sorted(edges)}
            development = dict(schema='HGB_SOURCE_TRAIN_VAL_ONLY', source_member='ACM/label.dat',
                               source_member_sha256=hashes['ACM/label.dat'], target_type=target_type,
                               labels={str(n): pool[n] for n in sorted(pool)})
            development_path = run / 'DEVELOPMENT_LABELS.json'
            development_path.write_text(json.dumps(development, indent=2, sort_keys=True) + '\n')
            row.update(status='official_archive_and_schema_acquired', member_sha256=hashes,
                       node_types=types, relations=relations, total_nodes=len(nodes), raw_edge_total=sum(edges.values()),
                       development_pool=dict(nodes=len(pool), class_counts=dict(classes), target_type=target_type),
                       development_labels=dict(path=str(development_path), bytes=development_path.stat().st_size,
                                               sha256=hashlib.sha256(development_path.read_bytes()).hexdigest()),
                       opened_archive_payloads=['ACM/node.dat', 'ACM/link.dat', 'ACM/label.dat'])
    except Exception as error:
        row.update(status='acquisition_or_schema_failed', error_type=type(error).__name__,
                   error_message=str(error), traceback=traceback.format_exc())
    (run / 'ACQUISITION_SCHEMA.json').write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    print(json.dumps(row, sort_keys=True))
    return 0 if row['status'] == 'official_archive_and_schema_acquired' else 1


if __name__ == '__main__':
    raise SystemExit(main())
