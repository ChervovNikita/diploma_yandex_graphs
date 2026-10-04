"""Bind a separate source audit without reading project code or primary papers."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UTC = datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
            'sha256': sha(p)}


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False,
                                      sort_keys=True, indent=2) + '\n')


audit = ROOT / 'conditional_auxiliary_label_semantics_audit_20261004_v1'
audit_manifest = json.loads((audit / 'MANIFEST.json').read_text())
audit_seal = json.loads((audit / 'SEAL.json').read_text())
assert sha(audit / 'MANIFEST.json') == audit_seal['manifest_sha256']
assert (audit / 'MANIFEST.json').stat().st_size == audit_seal['manifest_bytes']
audit_refs = [ref(audit / name) for name in
              ['MEMO.md', 'MANIFEST.json', 'SEAL.json']]
amendment = ROOT / 'conditional_pattern_serving_alignment_note_20261004_v1/SERVING_BOUNDARY_AMENDMENT.md'
save('ADDITIONAL_SOURCE_AUDIT_CONTEXT.json', {
    'UTC': UTC,
    'status': 'Cached source-audit conclusions adopted with exact scope limits',
    'consulted_cached_sources': audit_refs + [ref(amendment)],
    'source_audit_manifest_binding_verified': True,
    'new_project_source_semantic_reads': 0,
    'new_primary_paper_reads': 0,
    'serving_fact': 'On unmasked complete TRAIN, current residual teacher subsets have kL=kR=0; all conditional member likelihoods are one, responsibilities uniform, qJ/qF=1.',
    'negative_admission_fact': 'The sealed Collab/DDI admitted streams reject TRAIN-positive negative collisions; this does not certify latent or future absence or product-law sampling.',
    'scope_boundary': 'Source/mask/native-sampler claims come from the separate sealed audit memo, not further code auditing here. Current studies remain training-only auxiliary with original mean serving.',
    'successor_requirement': 'A nontrivial structural serving gate needs a separately fixed inference mask/view/support and observable reconstruction protocol with paid graph passes and likelihood work. No protocol or execution is adopted.'
})

marker = '\nSEPARATE SOURCE-AUDIT QUALIFICATION\n'
report_path = HERE / 'REPORT.txt'
report = report_path.read_text().split(marker)[0]
report += marker + '\n'
report += ('The separately sealed conditional_auxiliary_label_semantics_audit_20261004_v1/MEMO.md and the preserved serving-alignment amendment qualify the current setup. Their conclusions are reused here; no additional project-source semantic read was performed. The admitted Collab and DDI negative streams reject TRAIN-positive collisions, although such queries remain unobserved rather than verified latent/future nonedges or exact qF samples.\n\n'
           'On unmasked complete TRAIN, the current residual teacher-positive subsets are empty: k_L=k_R=0 even when candidate supports are nonempty. Every conditional member likelihood is one, both responsibility vectors are uniform, and qJ/qF=1. Removing only the query edge retains this limit once both endpoints are excluded. Thus the present conditional teacher evidence cannot provide a nontrivial served gate. Native completion predictions do not change this teacher-law result. A successor needs a separately fixed inference mask/view/support and observable reconstruction protocol, with paid graph passes and likelihood work and a declared train/serve shift. This is a necessary boundary, not an adopted protocol or additional experiment. The existing studies retain their training-only auxiliary and original mean-score serving. Exact audit/amendment pins are saved in ADDITIONAL_SOURCE_AUDIT_CONTEXT.json.\n')
report_path.write_text(report)

hypothesis = json.loads((HERE / 'HYPOTHESIS_AND_LIMITS.json').read_text())
new_limit = 'Separate source audit: current full-TRAIN teacher counts are both zero, so qJ/qF=1; nontrivial serving requires a separately fixed, paid inference observation protocol.'
if new_limit not in hypothesis['limits']:
    hypothesis['limits'].append(new_limit)
hypothesis['additional_source_audit_context'] = 'ADDITIONAL_SOURCE_AUDIT_CONTEXT.json'
save('HYPOTHESIS_AND_LIMITS.json', hypothesis)
conclusions = json.loads((HERE / 'CONCLUSIONS.json').read_text())
conclusions['additional_source_audit_context'] = 'ADDITIONAL_SOURCE_AUDIT_CONTEXT.json'
conclusions['current_full_train_teacher_bayes_factor'] = 1
conclusions['current_teacher_law_nontrivial_posterior_serving_supported'] = False
save('CONCLUSIONS.json', conclusions)
verification = json.loads((HERE / 'VERIFICATION.json').read_text())
verification['report_word_count'] = len(report.split())
verification['source_audit_context_added_UTC'] = UTC
verification['source_audit_seal_manifest_binding_verified'] = True
verification['additional_project_source_semantic_reads'] = 0
verification['additional_primary_semantic_reads'] = 0
save('VERIFICATION.json', verification)

payload = sorted(p for p in HERE.rglob('*')
                 if p.is_file() and p.name not in ['MANIFEST.json', 'SEAL.json'])
save('MANIFEST.json', {'schema': 'sha256_manifest_v1', 'UTC': UTC,
     'files': [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size,
                'sha256': sha(p)} for p in payload],
     'file_count': len(payload), 'execution_authorized': False})
save('SEAL.json', {'schema': 'sha256_seal_v1', 'UTC': UTC,
     'manifest': ref(HERE / 'MANIFEST.json'), 'report': ref(report_path),
     'conclusions': ref(HERE / 'CONCLUSIONS.json'),
     'source_index': ref(ROOT / 'literature_memory/index_v50/LITERATURE_INDEX.json'),
     'execution_authorized': False})
print(json.dumps({'manifest_sha256': sha(HERE / 'MANIFEST.json'),
                  'seal_sha256': sha(HERE / 'SEAL.json'),
                  'report_word_count': len(report.split()),
                  'new_primary_reads': 0,
                  'new_project_source_semantic_reads': 0}, indent=2))
