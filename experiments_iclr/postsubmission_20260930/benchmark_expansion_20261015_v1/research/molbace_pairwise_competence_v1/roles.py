"""TRAIN-only symmetric input-pair roles; stdlib construction and scalar checks.

No model, tensor library, graph payload or source record is loaded at import.
"""
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
import random


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def cosine(a,b):
    # Both operands have already been normalized in Python float64; no cutoff.
    return math.fsum(x*y for x,y in zip(a,b))


def rule(scores):
    """Three empirical boundaries, with exact integer tied-mass numerators.

    Every input score counts once, independent of labels. A tied group occupies
    [4L,4R] and quartile k occupies [kN,(k+1)N] in quarter-rank units.
    """
    values=sorted(scores);n=len(values)
    require(n>0 and all(math.isfinite(v) for v in values),'Finite distinct-input pair scores required')
    boundaries=[values[(k*n+3)//4-1] for k in (1,2,3)]
    tied=[]
    for value in sorted(set(boundaries)):
        left,right=bisect_left(values,value),bisect_right(values,value)
        numerator=[max(0,min(4*right,(k+1)*n)-max(4*left,k*n)) for k in range(4)]
        tied.append({'score':value,'numerators':numerator,'denominator':4*(right-left)})
    unique=largest=run=0;previous=None
    for value in values:
        if value!=previous:unique+=1;run=0;previous=value
        run+=1;largest=max(largest,run)
    return {'schema':'TRAIN-input-similarity-fractional-quartiles-v1','all_unordered_pairs':n,
        'boundaries':boundaries,'boundary_ties':tied,'labels_used':False,
        'minimum_similarity':values[0],'maximum_similarity':values[-1],
        'unique_similarity_scores':unique,'largest_tied_group':largest}


def membership(frozen,score):
    for row in frozen['boundary_ties']:
        if score==row['score']:return tuple(v/row['denominator'] for v in row['numerators'])
    selected=sum(score>b for b in frozen['boundaries'])
    return tuple(float(k==selected) for k in range(4))


def weights(frozen,score):
    return tuple(.5+2*q for q in membership(frozen,score))


def support_distribution(positive,negative,missing,batch):
    """Exact class-count hypergeometric law, no sampler/RNG/model access."""
    total=positive+negative+missing;denominator=math.comb(total,batch);histogram=Counter()
    for p in range(min(positive,batch)+1):
        for n in range(min(negative,batch-p)+1):
            m=batch-p-n
            if 0<=m<=missing:histogram[p*n]+=math.comb(positive,p)*math.comb(negative,n)*math.comb(missing,m)
    require(sum(histogram.values())==denominator,'Exact minibatch support law must have unit mass')
    expected=positive*negative*batch*(batch-1)/(total*(total-1))
    return {'batch_size':batch,'empty_pair_probability':histogram[0]/denominator,
        'both_class_probability':1-histogram[0]/denominator,'expected_observed_pairs':expected,
        'pair_support_distribution':[{'observed_pairs':k,'probability':v/denominator} for k,v in sorted(histogram.items())],
        'scope':'Uniform without-replacement minibatch class counts before native skip law; no RNG, score, dropout or independence claim.'}


class Table:
    def __init__(self,ids,histograms,labels,*,variant,seed):
        require(variant in ('common','structured','permuted') and type(seed)is int,'Frozen role variant/seed required')
        require(len(ids)==len(histograms)==len(labels)>1 and len(set(ids))==len(ids) and
            all(type(i)is int for i in ids),'Unique authentic TRAIN graph IDs required')
        order=sorted(range(len(ids)),key=lambda i:ids[i]);self.ids=[ids[i] for i in order]
        self.labels=[labels[i] for i in order]
        require(all(y in (None,0,1) for y in self.labels),'Observed binary labels or missing required')
        width=len(histograms[0]);self.normalized=[]
        for index in order:
            row=histograms[index]
            require(len(row)==width>0 and all(math.isfinite(v) and v>=0 for v in row),'Native finite nonnegative B required')
            norm=math.sqrt(math.fsum(v*v for v in row))
            self.normalized.append(tuple(v/norm for v in row) if norm else tuple(0. for _ in row))
        self.position={i:k for k,i in enumerate(self.ids)};self.variant=variant;self.seed=seed
        # All unordered distinct TRAIN inputs, including same/missing-label pairs.
        scores=array('d',(cosine(a,b) for i,a in enumerate(self.normalized) for b in self.normalized[i+1:]))
        self.frozen=rule(scores)
        score_sha=hashlib.sha256(scores.tobytes()).hexdigest()
        self.positive=[i for i,y in enumerate(self.labels) if y==1]
        self.negative=[i for i,y in enumerate(self.labels) if y==0]
        require(self.positive and self.negative,'Real MolBACE TRAIN must support both classes')
        self.pos_order={i:k for k,i in enumerate(self.positive)};self.neg_order={i:k for k,i in enumerate(self.negative)}
        # At most four pure quartiles plus three boundary-tie vectors. The
        # permutation stores a byte code per observed positive-negative pair.
        palette=[];codes=array('B')
        for i in self.positive:
            for j in self.negative:
                vector=weights(self.frozen,cosine(self.normalized[i],self.normalized[j]))
                if vector not in palette:palette.append(vector)
                codes.append(palette.index(vector))
        self.palette=tuple(palette);before=Counter(codes)
        self.codes=array('B',codes)
        if variant=='permuted':random.Random(seed+7030031).shuffle(self.codes)
        require(Counter(self.codes)==before,'Whole-vector permutation multiset changed')
        self._summary=self._summarize(scores,score_sha,before)

    def _pair_index(self,i,j):
        if self.labels[i]==0:i,j=j,i
        require(self.labels[i]==1 and self.labels[j]==0 and i!=j,'Observed distinct opposite-label TRAIN pair required')
        return self.pos_order[i]*len(self.negative)+self.neg_order[j]

    def vector(self,first,second):
        i,j=self.position[first],self.position[second]
        key=self._pair_index(i,j)
        return (1.,)*4 if self.variant=='common' else self.palette[self.codes[key]]

    def batch(self,ids,labels):
        require(len(ids)==len(labels) and len(set(ids))==len(ids),'Distinct aligned native minibatch IDs required')
        for i,y in zip(ids,labels):
            require(i in self.position and y==self.labels[self.position[i]],'Batch identity/label differs from frozen TRAIN')
        positive=[i for i,y in enumerate(labels) if y==1];negative=[i for i,y in enumerate(labels) if y==0]
        pairs=[(i,j) for i in positive for j in negative]
        return pairs,[self.vector(ids[i],ids[j]) for i,j in pairs]

    def _summarize(self,scores,score_sha,before):
        n=len(self.ids);counts=[0]*n;mass=[[0.]*4 for _ in range(n)]
        for i in self.positive:
            for j in self.negative:
                vector=self.vector(self.ids[i],self.ids[j]);counts[i]+=1;counts[j]+=1
                for m,w in enumerate(vector):mass[i][m]+=w;mass[j][m]+=w
        def group(indices):
            return {'molecules':len(indices),'pair_support_per_molecule':{
                'minimum':min((counts[i] for i in indices),default=0),'maximum':max((counts[i] for i in indices),default=0)},
                'route_pair_mass':[math.fsum(mass[i][m] for i in indices) for m in range(4)],
                'per_molecule_route_mass':[{'minimum':min((mass[i][m] for i in indices),default=0.),
                    'maximum':max((mass[i][m] for i in indices),default=0.),
                    'mean':math.fsum(mass[i][m] for i in indices)/len(indices) if indices else None} for m in range(4)]}
        all_mass=[math.fsum(weights(self.frozen,v)[m] for v in scores) for m in range(4)]
        expected=len(scores)
        require(all(math.isclose(v,expected,rel_tol=1e-12,abs_tol=1e-9) for v in all_mass),'All-input route mass is not balanced')
        pair_count=len(self.codes)
        missing=sum(y is None for y in self.labels)
        batch_sizes=sorted(set((min(n,32),n%32 or min(n,32))))
        route_mass=[float(pair_count)]*4 if self.variant=='common' else [
            math.fsum(count*self.palette[code][m] for code,count in before.items()) for m in range(4)]
        return {'schema':'molbace-TRAIN-symmetric-pair-role-table-v1','variant':self.variant,'TRAIN_inputs':n,
            'TRAIN_ids_sha256':digest(self.ids),'TRAIN_labels_sha256':digest(self.labels),
            'normalized_B_sha256':digest(self.normalized),'unordered_similarity_FP64_sha256':score_sha,
            'zero_bond_inputs':sum(not any(v) for v in self.normalized),'rule':self.frozen,
            'all_input_structured_route_mass':all_mass,'all_input_pair_support':len(scores),
            'observed_positive_inputs':len(self.positive),'observed_negative_inputs':len(self.negative),
            'missing_label_inputs':missing,'observed_opposite_pair_support':pair_count,
            'analytic_minibatch_support':[support_distribution(len(self.positive),len(self.negative),missing,b) for b in batch_sizes],
            'opposite_label_route_mass':route_mass,'structured_opposite_pair_vector_palette':[list(v) for v in self.palette],
            'structured_opposite_pair_vector_counts':dict(sorted(before.items())),'assigned_pair_code_sha256':hashlib.sha256(self.codes.tobytes()).hexdigest(),
            'role_codes_used':self.variant!='common',
            'permutation_seed':self.seed+7030031 if self.variant=='permuted' else None,
            'permutation_order':'positive source graph ID ascending, then negative source graph ID ascending',
            'whole_vector_multiset_preserved':True,'per_class_mass':{'positive':group(self.positive),'negative':group(self.negative),
                'missing':group([i for i,y in enumerate(self.labels) if y is None])},
            'per_molecule_support_and_mass_sha256':digest({'ids':self.ids,'pair_support':counts,'route_mass':mass}),
            'retained_pair_code_bytes':len(self.codes),'VALID_inputs_used':0,'model_forwards':0,
            'mass_limit':'Equal mass over all input pairs does not imply equal opposite-label, molecule or class mass.'}

    def metadata(self):
        # Copy through JSON also canonicalizes integer count keys for restore.
        return json.loads(json.dumps(self._summary,sort_keys=True,allow_nan=False))
