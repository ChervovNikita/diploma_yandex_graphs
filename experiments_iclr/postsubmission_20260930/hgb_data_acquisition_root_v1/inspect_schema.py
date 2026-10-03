"""Inspect complete released DBLP features/topology and development label pool.

No heldout label member is opened and no split or model is selected here.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()==str(REPO)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=REPO/'experiments_iclr/postsubmission_20260930/hgb_data_acquisition_root_v1'
archive=root/'download01/DBLP.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='0d3ea4a74399f9cd3e83af206e8e0b67e1844fe2c8463b424189884dd58ad7c8'
out=root/'schema01';out.mkdir(exist_ok=False)
hashes={}

def lines(z,name):
    digest=hashlib.sha256()
    with z.open(name) as handle:
        for raw in handle:
            digest.update(raw)
            yield raw.decode().rstrip('\r\n').split('\t')
    hashes[name]=digest.hexdigest()


with zipfile.ZipFile(archive) as z:
    nodes={};counts=Counter();widths=defaultdict(Counter);absent=Counter();nodeids=defaultdict(list)
    for parts in lines(z,'DBLP/node.dat'):
        assert len(parts) in (3,4), len(parts)
        nid,nt=int(parts[0]),int(parts[2]);assert nid not in nodes
        nodes[nid]=nt;counts[nt]+=1;nodeids[nt].append(nid)
        if len(parts)==3 or not parts[3]: absent[nt]+=1
        else:
            values=parts[3].split(',');widths[nt][len(values)]+=1
            assert all(math.isfinite(float(v)) for v in values)
    assert len(nodes)==26128 and set(nodes)==set(range(26128))
    types={}
    for nt in sorted(counts):
        ids=sorted(nodeids[nt]);assert ids==list(range(min(ids),max(ids)+1))
        assert len(widths[nt])<=1
        types[nt]=dict(nodes=counts[nt],global_id_offset=min(ids),feature_width_counts=dict(widths[nt]),absent_feature_rows=absent[nt])
    seen=defaultdict(set);rawcounts=Counter();weightcounts=defaultdict(Counter);endpoints={};selfcounts=Counter()
    for parts in lines(z,'DBLP/link.dat'):
        assert len(parts)==4
        src,dst,rel=int(parts[0]),int(parts[1]),int(parts[2]);w=float(parts[3])
        assert src in nodes and dst in nodes and math.isfinite(w)
        pair=(nodes[src],nodes[dst]);assert rel not in endpoints or endpoints[rel]==pair
        endpoints[rel]=pair;rawcounts[rel]+=1;seen[rel].add((src,dst));weightcounts[rel][w]+=1
        selfcounts[rel]+=src==dst
    assert sum(rawcounts.values())==479132 and len(rawcounts)==6
    relations={rel:dict(source_type=endpoints[rel][0],target_type=endpoints[rel][1],raw_records=rawcounts[rel],unique_directed_pairs=len(seen[rel]),duplicate_records=rawcounts[rel]-len(seen[rel]),raw_weight_counts=dict(weightcounts[rel]),self_records=selfcounts[rel]) for rel in sorted(rawcounts)}
    pool={};classes=Counter()
    for parts in lines(z,'DBLP/label.dat'):
        assert len(parts)==4
        nid,nt=int(parts[0]),int(parts[2]);labels=parts[3].split(',')
        assert nid in nodes and nodes[nid]==nt==0 and len(labels)==1 and nid not in pool
        label=int(labels[0]);assert 0<=label<4
        pool[nid]=label;classes[label]+=1
    receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),status='complete_schema_verified',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),member_sha256=hashes,node_types=types,relations=relations,development_pool=dict(nodes=len(pool),class_counts=dict(classes)),native_HGT_feature_type2=dict(target_width=next(iter(widths[0])) if widths[0] else counts[0],other_types_sparse_identity={nt:counts[nt] for nt in sorted(counts) if nt!=0}),scope=dict(full_features_topology_parsed=True,TRAIN_VAL_pool_parsed=True,heldout_label_payload_opened=False,public_info_meta_previously_opened=True,split_adopted=False,model_or_optimizer_execution=False,original_scores_recalculated=False))
    (out/'SCHEMA.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
