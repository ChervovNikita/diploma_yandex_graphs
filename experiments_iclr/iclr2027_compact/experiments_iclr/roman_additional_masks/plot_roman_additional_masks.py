"""Display every paired result from the independently audited Roman48 grid."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
j = json.loads(a.audit.read_text())
assert j['status'] == 'ROOT_INDEPENDENT_48_DECISION_LABEL_TRACE_AUDIT_PASS'
rows = {(r['mask'], r['depth'], r['seed'], r['arm']): r for r in j['records']}
assert set(rows) == {(m,d,s,arm) for m in range(1,5) for d in (2,5) for s in range(3) for arm in ('tied','untied_propagation')}
diff = {(m,d,s):100*(rows[m,d,s,'tied']['test_accuracy']-rows[m,d,s,'untied_propagation']['test_accuracy'])
        for m in range(1,5) for d in (2,5) for s in range(3)}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1,4,figsize=(10.4,3.3),sharey=True,constrained_layout=True)
for mask, ax in zip(range(1,5),axes):
    for seed in range(3):
        ax.plot([2,5],[diff[mask,d,seed] for d in (2,5)],'o-',
                color='#8ba3b0',alpha=.65,linewidth=.9,markersize=4)
    ax.scatter([2,5],[np.mean([diff[mask,d,s] for s in range(3)]) for d in (2,5)],
               marker='D',s=50,color='#204e68',zorder=5)
    ax.axhline(0,color='#444444',linewidth=.8)
    ax.set_xticks([2,5]); ax.set_xlim(1.5,5.5)
    ax.set_title(f'Official mask {mask}')
    ax.set_xlabel('SAGE depth'); ax.grid(axis='y',color='#dddddd',linewidth=.5)
axes[0].set_ylabel('TIED minus UNTIED accuracy\n(percentage points)')
fig.suptitle('Additional Roman masks, width 128, 1,000 epochs, no added loops',fontsize=12)
a.output_dir.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output_dir/'roman_additional_masks.png',dpi=220,bbox_inches='tight')
plt.close(fig)
out=[r'\begin{table}[t]',r'\centering',
     r'\caption{All additional-mask Roman results. Entries are TIED-minus-UNTIED test accuracy in percentage points. Seeds repeat optimization within each mask. Masks overlap on the same graph.}',
     r'\label{tab:roman-additional-masks}',r'\small',
     r'\begin{tabular}{rrrrrr}',r'\toprule',
     r'Mask & Depth & Seed 0 & Seed 1 & Seed 2 & Mean\\',r'\midrule']
for m in range(1,5):
    for d in (2,5):
        values=[diff[m,d,s] for s in range(3)]
        out.append(f'{m} & {d} & '+' & '.join(f'${v:+.3f}$' for v in [*values,float(np.mean(values))])+r'\\')
out += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(a.output_dir/'roman_additional_masks_table.tex').write_text('\n'.join(out)+'\n')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
(a.output_dir/'roman_additional_masks_figure_provenance.json').write_text(json.dumps({
    'audit_sha256':sha(a.audit),'generator_sha256':sha(Path(__file__)),
    'source_freeze_sha256':j['source_freeze_sha256'],
    'scope':'All24 paired differences. Lines connect the same mask/seed at two depths, diamonds are means. No across-graph inference.',
    'figure_sha256':sha(a.output_dir/'roman_additional_masks.png')},indent=2)+'\n')
for d in (2,5):
    epochs=[r['selected_epoch'] for r in rows.values() if r['depth']==d]
    print('depth',d,'selected epoch range',min(epochs),max(epochs),'after900',sum(e>900 for e in epochs),'of',len(epochs))
