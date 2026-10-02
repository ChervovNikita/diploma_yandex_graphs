"""Metadata-only graph-name exposure search: emits names/paths/hashes, no outcomes."""
from pathlib import Path
import json,re,hashlib,datetime
ROOT=Path(__file__).resolve().parent
PHASE=ROOT.parents[1]
assert not (ROOT/'SEAL.json').exists()
names={'Squirrel':r'\bsquirrel\b','PubMed':r'\bpubmed\b','DeezerEurope':r'deezer[ _-]?europe','Penn94':r'\bpenn[ _-]?94\b'}
candidates=[]
# Restrict to explicit source/launch/protocol/context metadata documents.
# Never traverse scientific result payloads, datasets, tensors or model artifacts.
for p in PHASE.rglob('*'):
 if not p.is_file() or p.suffix.lower() not in ('.md','.json') or ROOT in p.parents:continue
 rel=p.relative_to(PHASE)
 if any(part.lower() in ('results','primary','renders','discovery','evidence','data','datasets','models','replays','science_authorized1gpu_run01') for part in rel.parts):continue
 name=p.name.lower()
 if name in ('context.md',) or ('launch' in name and 'json'==p.suffix[1:].lower()) or ('protocol' in name and p.suffix=='.md') or ('metadata' in name and p.suffix=='.json'):
  candidates.append(p)
records=[]
for p in sorted(candidates):
 data=p.read_bytes()
 try:body=data.decode('utf-8')
 except UnicodeDecodeError:continue
 hits=[name for name,pattern in names.items() if re.search(pattern,body,re.I)]
 if hits:records.append(dict(path=str(p.relative_to(PHASE)),sha256=hashlib.sha256(data).hexdigest(),graph_names=hits,meaning='Graph-name mention in metadata only; not proof of an executed or untuned graph.'))
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Metadata graph-name search only; no numeric outcomes returned or scientific artifact accessed.',candidate_document_count=len(candidates),matched_documents=records,limits='Metadata mention does not distinguish a proposed route from executed/tuned exposure. No absence claim outside these eligible phase metadata records or earlier legacy studies. Exact prior-exposure status remains unresolved until authoritative launch/exposure provenance resolves it.',decision='Provisional Squirrel/PubMed cannot be frozen/acquired as prospectively new tasks from original-five exclusion alone.')
(ROOT/'GRAPH_EXPOSURE_METADATA.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
