"""Recompute all Cora preprocessing groups and default decision accounting."""
from pathlib import Path
import argparse, csv, json
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=['base','ens','tied','private_first','private_last','untied']
NAMES=['BASE','ENS','TIED','Private first','Private last','UNTIED']

def compute():
    rows=list(csv.DictReader((ROOT/'summary/selected_default.csv').open()))
    assert len(rows)==24
    records=[]
    for r in rows:
        ds=r['dataset'];a=r['arm'];c=f"lr{float(r['lr']):g}_wd{float(r['weight_decay']):g}"
        with np.load(ROOT/f'hard_decisions/{ds}/test_reference.npz') as z:
            truth=z['full_labels'][z['test_indices']]
        p=[];m=[]
        for seed in range(3):
            with np.load(ROOT/f'hard_decisions/{ds}/{a}/{c}/seed{seed}.npz') as z:
                p.append(float(100*np.mean(z['pooled_test_class']==truth)))
                m.append(float(100*np.mean(z['member_test_class']==truth)))
            assert abs(p[-1]-100*float(r[f'test_seed{seed}']))<1e-5
        records.append({'dataset':ds,'arm':a,'role':r['role'],'pooled_pp':p,'member_pp':m,
                        'mean_pooled_pp':float(np.mean(p)),'sd_pooled_pp':float(np.std(p,ddof=1)),
                        'mean_member_pp':float(np.mean(m)),
                        'gain_pp':float(np.mean(p)-np.mean(m))})
    index={(r['dataset'],r['arm'],r['role']):r for r in records}
    contrasts=[]
    for a in ARMS:
        raw=index['cora_raw',a,'default'];norm=index['cora_normalized',a,'default']
        values={name:norm[k]-raw[k] for name,k in [('pooled_delta_pp','mean_pooled_pp'),('member_delta_pp','mean_member_pp'),('gain_delta_pp','gain_pp')]}
        assert abs(values['pooled_delta_pp']-values['member_delta_pp']-values['gain_delta_pp'])<1e-12
        contrasts.append({'arm':a,**values})
    return {'scope':'All selected/default groups and all fixed-default normalization differences on one Cora public split. Descriptive held-out decision accounting.',
            'records':records,'default_normalization_contrasts':contrasts}

def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1.1,1]})
    index={(r['dataset'],r['arm'],r['role']):r for r in result['records']}
    y=np.arange(6)
    for ds,off,color,label in [('cora_raw',-.14,'#757575','Raw features'),('cora_normalized',.14,'#176EAB','Row-normalized features')]:
        groups=[index[ds,a,'selected'] for a in ARMS]
        for i,r in enumerate(groups):ax.scatter(r['pooled_pp'],[i+off]*3,s=13,color=color,alpha=.42)
        ax.scatter([r['mean_pooled_pp'] for r in groups],y+off,marker='D',s=27,color=color,label=label)
    ax.set_yticks(y,NAMES);ax.invert_yaxis();ax.set_xlabel('Validation-selected test accuracy (%)')
    ax.set_title('(a) Same graph, two feature conditions',loc='left',fontweight='bold',fontsize=10)
    ax.grid(axis='x',alpha=.18);ax.legend(loc='lower left',fontsize=8,frameon=False)
    rows=result['default_normalization_contrasts']
    member=np.array([r['member_delta_pp'] for r in rows]);gain=np.array([r['gain_delta_pp'] for r in rows]);pooled=member+gain
    bx.barh(y-.17,member,height=.3,color='#176EAB',label='Member accuracy change')
    bx.barh(y+.17,gain,height=.3,color='#D1862A',label='Pooling-gain change')
    bx.scatter(pooled,y,s=25,marker='D',color='#242424',label='Pooled accuracy change',zorder=3)
    bx.axvline(0,color='#888888',lw=.8);bx.set_yticks(y,NAMES);bx.invert_yaxis()
    bx.set_xlabel('Normalized minus raw (percentage points)')
    bx.set_title('(b) Common default optimizer',loc='left',fontweight='bold',fontsize=10)
    bx.grid(axis='x',alpha=.18);bx.legend(loc='lower left',fontsize=7,frameon=False)
    fig.tight_layout(w_pad=2.0);fig.savefig(path,dpi=240,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');parser.add_argument('--figure',type=Path);args=parser.parse_args()
    result=compute();target=ROOT/'PREPROCESSING_DECISION_ACCOUNTING.json'
    if args.write:target.write_text(json.dumps(result,indent=2)+'\n')
    else:assert result==json.loads(target.read_text())
    if args.figure:plot(result,args.figure)
    print('PASS: all24 groups and six fixed-default contrasts')
