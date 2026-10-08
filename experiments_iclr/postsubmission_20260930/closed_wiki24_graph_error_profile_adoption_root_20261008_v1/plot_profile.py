"""Render the full fixed degree profile as scientific PNG/SVG figures, never PDF."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert 'experiments_iclr/postsubmission_20260930' in str(directory)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    summary = json.loads((directory / 'SUMMARY.json').read_text())
    profile = json.loads((directory / 'PROFILE.json').read_text())
    bins = [s for s in summary['summaries'] if s['kind'] == 'degree_quartile']
    assert len(bins) == 4
    labels = ['≤4', '5–11', '12–44', '>44']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True)
    for ax, key, title in zip(axes,
                             ['member_difference_pp', 'coverage_difference_pp', 'served_difference_pp'],
                             ['Mean member accuracy', 'At least one correct member', 'Served accuracy']):
        values = np.array([s[key] for s in bins])
        means = values.mean(axis=1)
        colors = ['#2467a8' if x >= 0 else '#b6423b' for x in means]
        ax.bar(range(4), means, color=colors, width=.62, alpha=.75)
        for seed in range(3):
            ax.scatter(np.arange(4) + (seed - 1) * .09, values[:, seed],
                       marker=['o', 's', '^'][seed], color='#222222', s=24,
                       label=str(summary['seeds'][seed]), zorder=3)
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set_title(title)
        ax.set_xticks(range(4), labels)
        ax.set_xlabel('Incoming unique nonself neighbours')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Ordinary minus shared unit+contrast (pp)')
    axes[2].legend(title='Optimizer seed', frameon=False, fontsize=8)
    fig.suptitle('Complete Wiki24 degree profile: individual quality and useful alternatives', y=.99)
    fig.text(.5, .01, 'Bars: mean of three seeds. Points: every seed. '
             'Development-selected states on one split. No confidence intervals or causal claim.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(directory / ('degree_profile.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    classes = [None, *range(10)]
    matrix = np.zeros((11, 4), dtype=float)
    support = np.zeros((11, 4), dtype=int)
    for i, c in enumerate(classes):
        for q in range(1, 5):
            rows = [next(r for r in s['rows'] if r['cohort_kind'] == 'degree_quartile'
                         and r['cohort_value'] == q and r['truth_class'] == c)
                    for s in profile['seeds']]
            assert len({r['nodes'] for r in rows}) == 1
            support[i, q-1] = rows[0]['nodes']
            differences = [r['paired']['served_accuracy_change'] for r in rows]
            matrix[i, q-1] = 100 * np.mean(differences) if rows[0]['nodes'] else np.nan
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()), 1.0)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap='RdBu', vmin=-limit, vmax=limit,
                      aspect='auto')
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(11), ['All classes', *[f'Class {c}' for c in range(10)]])
    ax.set_xlabel('Incoming unique nonself neighbours')
    for i in range(11):
        for j in range(4):
            if support[i, j]:
                value = matrix[i, j]
                color = 'white' if abs(value) > .55 * limit else '#111111'
                text = f'{value:+.2f} pp\nn={support[i,j]}'
            else:
                color, text = '#333333', 'empty\nn=0'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    ax.set_title('Served accuracy difference by degree and class\n'
                 'Ordinary independent ensemble minus shared unit+contrast')
    colorbar = fig.colorbar(image, ax=ax, shrink=.8)
    colorbar.set_label('Mean paired difference over three seeds (pp)')
    fig.text(.5, .015, 'Every fixed class/degree cell is shown. n is nodes per seed. '
             'Selected development data. Small cells and class composition limit interpretation.',
             ha='center', fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ('png', 'svg'):
        fig.savefig(directory / ('class_degree_profile.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(figures=4, PDF_created=False, fitted_models=0,
                          all_fixed_cells_shown=True, data_changed=False)))


if __name__ == '__main__':
    main()
