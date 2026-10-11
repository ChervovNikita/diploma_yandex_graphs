"""Draw selected-development evidence from the completed immutable compact F tables."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H=Path(__file__).resolve().parent
SOURCE=H.parent/'SAGE_matched_joint_reference_77_closed_history_20261011_v1'

def main():
    with (SOURCE/'ARM_METRICS.csv').open() as f:arms={r['arm']:r for r in csv.DictReader(f)}
    with (SOURCE/'PAIRED_CONTRASTS.csv').open() as f:contrasts=list(csv.DictReader(f))
    pair=next(r for r in contrasts if r['readout']=='raw' and r['candidate']=='shared4_F' and r['reference']=='genuine_factorized_I4_F')
    counts=[int(pair[k]) for k in ('coverage_delta','pooling_loss_delta','aggregation_only_delta','net_pooled_correct_delta')]
    assert counts==[225,141,1,85] and counts[0]-counts[1]+counts[2]==counts[3]
    order=['shared4_F','genuine_factorized_I4_F','joint_untied4_F','ordinary_M1_F','factorized_M1_F']
    labels=['Shared4','Independent4','Joint untied4','Ordinary single','Factorized single']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'semibold','svg.fonttype':'none','savefig.facecolor':'white'})
    fig,(left,right)=plt.subplots(1,2,figsize=(8.6,3.15),gridspec_kw={'width_ratios':[1.1,1.3]})
    color='#2A69AD';red='#C65456';gray='#6E7783'
    for y,(arm,label) in enumerate(zip(order,labels)):
        row=arms[arm];member=float(row['mean_member_accuracy_pct']);pooled=float(row['raw_accuracy_pct'])
        left.plot([member,pooled],[y,y],color=color if y==0 else gray,lw=1.4)
        left.scatter([member],[y],marker='^',s=38,facecolors='white',edgecolors=color if y==0 else gray,zorder=3)
        left.scatter([pooled],[y],marker='o',s=28,color=color if y==0 else gray,zorder=4)
        left.annotate(f'{pooled:.2f}',(pooled,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
    left.set_yticks(range(5),labels);left.invert_yaxis();left.set_xlim(77.8,81.2)
    left.set_xlabel('Accuracy (%)');left.set_title('(a) Individual and combined accuracy',loc='left',fontsize=10)
    left.grid(axis='x',alpha=.18);left.set_axisbelow(True)
    left.legend(handles=[Line2D([0],[0],marker='^',mfc='white',mec=gray,linestyle='',label='Mean member'),
        Line2D([0],[0],marker='o',color=gray,linestyle='',label='Served prediction')],loc='lower left',bbox_to_anchor=(0,-.39),
        frameon=False,ncol=2,fontsize=8,handletextpad=.4,columnspacing=1.2)
    deltas=[counts[0],-counts[1],counts[2]];starts=[0,counts[0],counts[0]-counts[1]]
    for i,(start,delta) in enumerate(zip(starts,deltas)):
        right.bar(i,abs(delta),bottom=min(start,start+delta),color=color if delta>0 else red,width=.65,zorder=3)
        end=start+delta
        right.text(i,max(start,end)+8,f'{delta:+d}',ha='center',fontsize=9,color=color if delta>0 else red)
        right.plot([i+.325,i+.675],[end,end],color=gray,lw=.8,alpha=.7)
    right.bar(3,counts[3],color='#21486D',width=.65,zorder=3)
    right.text(3,counts[3]+8,f'{counts[3]:+d}',ha='center',fontsize=9,color='#21486D')
    right.axhline(0,color=gray,lw=.8);right.set_ylim(0,265)
    right.set_xticks(range(4),['Added\ncoverage','Extra losses\nin averaging','Aggregation-only\ncorrection','Net correct\npredictions'],fontsize=8)
    right.set_ylabel('Change in correct-case counts')
    right.set_title('(b) Decision differences versus independent4',loc='left',fontsize=10)
    right.grid(axis='y',alpha=.18);right.set_axisbelow(True)
    fig.subplots_adjust(left=.155,right=.99,bottom=.25,top=.88,wspace=.33)
    fig.savefig(H/'F_member_coverage.png',dpi=220)
    fig.savefig(H/'F_member_coverage.svg')
    binding={'source_tables':{name:{'path':str(SOURCE/name),'sha256':hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()}
        for name in ('ARM_METRICS.csv','PAIRED_CONTRASTS.csv')},'plotted_arms':order,
        'plotted_counts':dict(zip(('coverage_added','extra_pooling_losses','aggregation_only_added','net_correct_added'),counts)),
        'scope':'Three paired optimizer seeds on the same5274 selected-development nodes;15822 dependent repeated readouts. No independent-node inference or unused confirmation.',
        'source_paired_mean_accuracy_pp':float(pair['mean_accuracy_delta_pp']),
        'source_descriptive_paired_95_interval_pp':[float(pair['accuracy_descriptive_CI_low_pp']),float(pair['accuracy_descriptive_CI_high_pp'])],
        'material_confidence_limit':'Mean calibrated NLL harm to independent4+.021448 exceeds the frozen.02 cap.',
        'new_predictions_or_scores':False,'model_data_checkpoint_or_TEST_reads':False,'matplotlib':matplotlib.__version__,
        'outputs':{name:{'bytes':(H/name).stat().st_size,'sha256':hashlib.sha256((H/name).read_bytes()).hexdigest()} for name in ('F_member_coverage.png','F_member_coverage.svg')}}
    (H/'FIGURE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps({'figure_created':True,'new_scores':False,'counts':counts,'outputs':binding['outputs']}))

if __name__=='__main__':main()
