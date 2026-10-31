"""One-shot document/source seal. Never imports scientific/prototype modules."""
from pathlib import Path
import datetime, hashlib, json
ROOT=Path(__file__).resolve().parent
PHASE=ROOT.parents[1]
INDEX=ROOT.parent/'ROUND_INDEX.md'
# Reject an already sealed packet before any mutation, including render flags.
if (ROOT/'SEAL.json').exists() or (ROOT/'MANIFEST.json').exists():
 raise SystemExit('Already sealed; no mutation allowed. Write a separate addendum.')
if '## Round5 executable aligned-member follow-up' in INDEX.read_text():
 raise SystemExit('Index entry already exists; inspect disposition without mutation.')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def verify(base, path, expected):
 assert sha(base/path)==expected, path
for row in json.loads((ROOT/'CONTEXT_BINDINGS.json').read_text()):
 verify(PHASE,row['path'],row['sha256'])
for row in json.loads((ROOT/'SOURCE_INTEGRATION_BINDINGS.json').read_text()):
 verify(PHASE,row['path'],row['sha256'])
static=json.loads((ROOT/'STATIC_SOURCE_INSPECTION.json').read_text())
for row in static['files']:
 verify(ROOT,row['path'],row['sha256'])
 assert row['syntax_parsed'] and not row['imported'] and not row['executed']
for row in json.loads((ROOT/'evidence/PASSAGE_INDEX.json').read_text()):
 for path_key,hash_key in [('pdf_path','pdf_sha256'),('source_text_path','source_text_sha256'),('passage_path','passage_sha256')]:
  verify(ROOT,row[path_key],row[hash_key])
for row in json.loads((ROOT/'evidence/CODE_PASSAGE_INDEX.json').read_text()):
 verify(PHASE,row['source_path'],row['source_sha256'])
 verify(ROOT,row['passage_path'],row['passage_sha256'])
render_path=ROOT/'renders/RENDER_INDEX.json'
renders=json.loads(render_path.read_text())
assert len(renders)==16
for row in renders:
 verify(ROOT,row['path'],row['sha256'])
 row['visually_inspected']=True
 row['inspection_basis']='Actually viewed in preceding authoring context; not newly rendered/automatically inferred.'
contacts=sorted((ROOT/'renders').glob('contact_*.png'))
assert len(contacts)==8
# Mutate only after all pre-seal integrity checks succeed.
render_path.write_text(json.dumps(renders,indent=2)+'\n')
(ROOT/'renders/CONTACT_INSPECTION_INDEX.json').write_text(json.dumps([
 dict(path=str(p.relative_to(ROOT)),sha256=sha(p),visually_inspected=True,
      inspection_basis='Eight contacts actually viewed in preceding authoring context') for p in contacts],indent=2)+'\n')
payload=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
         for p in sorted(ROOT.rglob('*')) if p.is_file()]
manifest=dict(scope='SOURCE_ONLY_NO_SCIENTIFIC_EXECUTION',files=payload,payload_file_count=len(payload))
with (ROOT/'MANIFEST.json').open('x') as handle: json.dump(manifest,handle,indent=2);handle.write('\n')
seal=dict(sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_only=True,scientific_execution=False,pilot_admitted=False,
 additional_primaries_relative_to_round5=5,new_distinct_candidates=0,existing_candidate_retained=1,
 source_python_files=3,static_ast_only=True,report_sha256=sha(ROOT/'REPORT.md'),
 complete_method_delta_sha256=sha(ROOT/'COMPLETE_METHOD_DELTA.md'),
 protocol_sha256=sha(ROOT/'PROSPECTIVE_PROTOCOL.md'),
 admission_template_sha256=sha(ROOT/'prototype/ADMISSION_TEMPLATE.json'),
 implementation_sha256={Path(row['path']).name:row['sha256'] for row in static['files']},
 manifest_sha256=sha(ROOT/'MANIFEST.json'),payload_file_count=len(payload),
 inspected_pdf_pages=16,inspected_contacts=8,manifest_mismatches=0)
with (ROOT/'SEAL.json').open('x') as handle: json.dump(seal,handle,indent=2);handle.write('\n')
for row in payload: verify(ROOT,row['path'],row['sha256'])
with INDEX.open('a') as handle:
 handle.write('\n## Round5 executable aligned-member follow-up\n\nCompleted and sealed: `round5_alignment_followup_v1/REPORT.md`, `COMPLETE_METHOD_DELTA.md`, `PROSPECTIVE_PROTOCOL.md`, and `prototype/correction_screen_driver.py`.\n\nFive additional full primaries relative to immutable Round5; zero originality, zero new distinct utility candidates; one existing candidate retained. Three unexecuted Python files provide an end-to-end exact original-wrapper teacher/correction screen. Recommend revised Squirrel + Photo, three paired distinct source splits/seeds, label-blind derived20/10/5/10/5/50 roles retaining every Squirrel published-test node in the final pool. Six primary same-teacher postprocessors, capable shuffled/marginal/HeAD/CF controls, full schedules/all unsuccessful attempts/costs and once-frozen final reporting. Scalar matched-member covariance loses class-specific relationships and is not a new uncertainty principle.\n\n'+str(len(payload))+' payload files,16 visually inspected source pages,8 contacts and zero manifest mismatches. AST syntax only; no data/model artifacts/imports/fits/scores/tests/GPU/SSH/scientific execution or pilot. Runtime/native-control/resource qualification and separate scientific admission remain future work. Stage1/original scores unchanged and unused as candidate inputs/selection.\n')
print(json.dumps(dict(payload_files=len(payload),total_files=len(payload)+2,manifest_mismatches=0,
 manifest_sha256=sha(ROOT/'MANIFEST.json'),seal_sha256=sha(ROOT/'SEAL.json'),
 report_sha256=seal['report_sha256'],protocol_sha256=seal['protocol_sha256'],
 implementation_sha256=seal['implementation_sha256']),indent=2))
