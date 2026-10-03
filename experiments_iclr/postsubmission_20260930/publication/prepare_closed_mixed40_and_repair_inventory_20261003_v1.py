"""Prepare an explicit publication inventory; do not commit or push here."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
PUB = 'publication/closed_mixed40_and_memory_repair_20261003_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
BRANCH = 'codex/postsubmission-research-20260930'
REMOTE = '''import pathlib,subprocess,json
r=pathlib.Path(%r)
def run(*args):return subprocess.run(args,cwd=r,capture_output=True,text=True,check=True).stdout.strip()
assert pathlib.Path(run('git','rev-parse','--show-toplevel'))==r
assert run('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert run('git','branch','--show-current')==%r
assert not run('git','diff','--cached','--name-only')
assert run('git','remote','get-url','origin')=='git@github.com:ChervovNikita/diploma_yandex_graphs.git'
print(json.dumps(dict(head=run('git','rev-parse','HEAD'),branch=run('git','branch','--show-current'),route_verified=True,no_staged_edits=True)))
''' % (REPO, BRANCH)


def main():
    output = PHASE / PUB
    output.mkdir(exist_ok=False)
    completed = subprocess.run(SSH + [shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE])],
                               capture_output=True, text=True, timeout=60)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), ssh_destination=LOGIN,
                   exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    (output / 'GIT_PRECHECK.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert completed.returncode == 0
    checked = json.loads(completed.stdout)
    old = PHASE / 'publication/replay_and_new_hypothesis_progress_20261003_v1/README_MAIN.md'
    text = old.read_text()
    a, b = text.index('The six native Amazon Ratings recipe fits'), text.index('\nFoRDE\'s explicit streamer')
    text = text[:a] + '''The six native Amazon Ratings recipe fits and replay remain complete, with no GNNM comparison result. The fixed authored Polynormer-r comparison preserves raw features, 200 local and 2500 global epochs, three paired blocks and 15 distinct fits. V6 bounded retention keeps selected local/final states, every epoch and decision, and retirement records. Source review and both physical runtime captures passed. Fresh complete numerical qualification is admitted before the 15 predictive fits; no trained GNNM outcome exists yet.

The complete shared/private HGT objective study now contains 40 evaluated cases and 15 required native secondary rows. All frozen primary and reverse-role practical improvement gates fail on ACM and DBLP. The declared extension is closed without tuning, calibration rescue or heldout promotion. [Complete results and exact decision](experiments_iclr/postsubmission_20260930/graph_mixed40_frozen_scientific_decision_20261003_v1/RESULTS_SUMMARY.md) preserve all paired vectors and descriptive uncertainty. TEST remains closed.

The frozen NCNC hypothesis compares reconstruction of whole TRAIN observation patterns with matched marginal supervision. The likelihood has GRAN ancestry. V3 numerical qualification passed; both complete-graph attempts failed memory caps before a completed J update, with all costs retained. V4 uses activation recomputation while preserving complete support, RNG, batches, native target/inference paths and scientific work. Independent source review and fresh value/gradient/RNG/update parity remain required before predictive J/F fits. A count-aware structural single has a sealed implementation, with review and runtime pending. No useful structural specialization, graph-specific predictive gain or methodological novelty is established.

BUDDY and NCNC base comparisons require complete family closure before scoring. Their checkpoints and runtime records remain on the authorized servers.
''' + text[b:]
    a = text.index('### Research update, 3 October 2026')
    text = text[:a] + '''### Research update, 3 October 2026

The Mixed40 extension failed its unchanged primary and reverse-role practical gates on both development graphs. The numerical audit includes all 40 selected models plus 15 native secondary rows. Earlier failed evaluator transports and schema diagnosis are retained.

Two full-graph NCNC pattern qualification attempts exceeded memory caps before a completed update. The sealed V4 recomputation repair preserves the scientific work and adds explicit saved-V3/V4 parity probes. Amazon V6 source review and runtime captures passed; fresh numerical qualification precedes admission of the original 15 fits. These engineering results establish no predictive winner.

Six new scoped primary method reads on coherent reconstruction and expert specialization are saved in the [literature report](experiments_iclr/postsubmission_20260930/ncnc_coherent_completion_recent_literature_scout_20261003_v1/REPORT.md). The canonical index remains at 168 conclusions across 119 paper identities until its next integration. Whole-pattern supervision, structured singles and expert routing have relevant prior work. Any contribution still requires capable controls, prospective paired replication, heldout confirmation and fresh independent manuscript review.

Original paper scores, unsuccessful experiments, reviews and research decisions are preserved. See [current status](experiments_iclr/postsubmission_20260930/PUBLIC_STATUS.md) and the research ledger.
'''
    (output / 'README_MAIN.md').write_text(text)
    roots = [
        'graph_mixed_block_closed_family_evaluation_preparation_20261003_v3',
        'mixed_closed_family_evaluation_source_review_20261003_v3',
        'graph_mixed40_complete_evaluation_execution_root_20261003_v3',
        'graph_mixed40_complete_evaluation_root_command_20261003_v3',
        'graph_mixed40_frozen_scientific_decision_20261003_v1',
        'graph_ncNC_structural_pattern_full_graph_normal_supervision_preparation_20261003_v2',
        'graph_ncNC_structural_pattern_full_graph_normal_supervision_independent_source_review_20261003_v2',
        'graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v2',
        'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4',
        'ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1',
        'ncnc_coherent_completion_recent_literature_scout_20261003_v1',
        'amazon_polynormer_paired_family_source_preparation_20261003_v6',
        'amazon_polynormer_v6_bounded_retention_forecast_independent_source_review_20261003_v1',
        'amazon_polynormer_v6_source_review_runtime_compatibility_receipt_20261003_v1',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/fit_schedule_resource_candidate_v2_v6_bounded_retention',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_execution_metadata_preparation_v1',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_runtime_cpu_v1',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_runtime_cuda0_v1',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_runtime_execution_receipts_20261003_v1',
        'coordination_snapshots/20261003_mixed40_negative_and_memory_repair_adoption_v1',
    ]
    suffixes = {'.py', '.json', '.md', '.txt', '.html', '.diff', '.patch', '.log', '.raw', '.sha256', '.csv', '.xml', '.sh'}
    sources = {
        'PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json',
        'coordination/close_mixed40_frozen_gate_20261003_v1.py',
        'coordination/adopt_mixed40_negative_and_memory_repair_20261003_v1.py',
        'publication/prepare_closed_mixed40_and_repair_inventory_20261003_v1.py',
        PUB + '/GIT_PRECHECK.json',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/V6_CONSUMER_RELEASE_v1.json',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_releases/runtime_cpu_v1.json',
        'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_releases/runtime_gpu_v1.json',
    }
    excluded = []
    for name in roots:
        root = PHASE / name
        assert root.is_dir(), name
        for p in root.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            relative = str(p.relative_to(PHASE))
            if p.suffix not in suffixes or p.stat().st_size >= 2_000_000:
                excluded.append(dict(path=relative, bytes=p.stat().st_size, reason='nontext_or_large_author_evidence'))
                continue
            assert not p.is_symlink()
            sources.add(relative)
    def row(name, target=None):
        p = PHASE / name
        data = p.read_bytes()
        return dict(source=name, target=target or 'experiments_iclr/postsubmission_20260930/' + name,
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    files = [row(PUB + '/README_MAIN.md', 'README.md')] + [row(n) for n in sorted(sources)]
    assert len(files) == len({r['target'] for r in files})
    inventory = dict(schema='exact_research_publication_inventory_v1',
                     expected_head=checked['head'], branch=BRANCH,
                     message='Record complete negative Mixed40 results and graph-pattern repair evidence',
                     files=files, remove=[], excludes=excluded)
    (output / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(inventory=PUB + '/INVENTORY.json', head=checked['head'],
                         files=len(files), bytes=sum(r['bytes'] for r in files),
                         large_or_binary_excluded=len(excluded))))


if __name__ == '__main__':
    main()
