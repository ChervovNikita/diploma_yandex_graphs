"""Predeclared complete public-graph double-edge swaps; no import-time numerics."""
import json
import random
import time
from source import bind
from stage_plan import REWIRE


def graph_sets(ordered_edges,n):
    pairs=[tuple(int(x) for x in edge) for edge in ordered_edges]
    if any(len(edge)!=2 or min(edge)<0 or max(edge)>=n for edge in pairs) or len(pairs)!=len(set(pairs)):
        raise ValueError('Complete simple ordered public graph without duplicates required')
    directed=set(pairs)
    if any((b,a) not in directed for a,b in directed):raise ValueError('Complete symmetric undirected graph required')
    self_edges={edge for edge in directed if edge[0]==edge[1]}
    edges={(a,b) for a,b in directed if a<b}
    degree=[0]*n
    for a,b in edges:degree[a]+=1;degree[b]+=1
    return edges,self_edges,degree


def swap_graph(ordered_edges,n=19717,deadline=None):
    original,self_edges,before=graph_sets(ordered_edges,n)
    if len(original)<2:raise ValueError('Complete graph cannot support predeclared swaps')
    bank=sorted(original);edges=set(original);rng=random.Random(REWIRE['double_edge_swap_seed'])
    requested=REWIRE['accepted_swaps_per_nonself_undirected_edge']*len(bank)
    cap=REWIRE['maximum_proposals_per_requested_swap']*requested
    accepted=proposals=0
    while accepted<requested:
        if proposals>=cap:raise RuntimeError('Fixed rewire proposal cap exhausted; preserve failure without alternate seed/graph')
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Fixed rewire generation deadline')
        i,j=rng.sample(range(len(bank)),2);a,b=bank[i];c,d=bank[j];proposals+=1
        if rng.getrandbits(1):a,b=b,a
        if rng.getrandbits(1):c,d=d,c
        if len({a,b,c,d})!=4:continue
        first=tuple(sorted((a,d)));second=tuple(sorted((c,b)))
        if first in edges or second in edges or first==second:continue
        edges.remove(bank[i]);edges.remove(bank[j]);edges.add(first);edges.add(second)
        bank[i],bank[j]=first,second;accepted+=1
    result=sorted(self_edges|edges|{(b,a) for a,b in edges})
    after,self_after,degree_after=graph_sets(result,n)
    if before!=degree_after or self_edges!=self_after or len(after)!=len(original) or after==original:
        raise ValueError('Exact degree/self-edge/count preservation and nontrivial topology required')
    return result,dict(double_edge_swap_seed=190121,swap_count=requested,rejection_cap=cap,
        proposals=proposals,accepted_swaps=accepted,nonself_undirected_edges=len(original),
        ordered_edges=len(result),nodes=n,self_edges=len(self_edges),isolated_nodes=sum(d==0 for d in before),
        exact_degree_sequence_preserved=True,exact_self_edges_preserved=True,
        retained_original_nonself_edges=len(original&after),
        retained_original_edge_fraction=len(original&after)/len(original),
        overlap_is_construction_metadata_not_selection=True,labels_used=False)


def admit_auxiliary(spec,bindings):
    graph=bind(spec['auxiliary_graph']);recipe=spec['frozen_rewire_recipe']
    custody=json.loads(bind(recipe['generation_custody']).read_text())
    if custody.get('schema')!='masked-context-degree-preserving-rewire-custody-v1' or custody.get('algorithm')!='double_edge_swap':
        raise ValueError('Exact separately frozen public graph construction custody required')
    if recipe.get('double_edge_swap_seed')!=190121 or custody.get('double_edge_swap_seed')!=190121:
        raise ValueError('One predeclared rewire seed; no graph selection or grid')
    for key in ('swap_count','rejection_cap','graph_fingerprint'):
        if custody.get(key)!=recipe.get(key):raise ValueError('Graph custody/recipe mismatch')
    E=custody.get('nonself_undirected_edges')
    if type(E) is not int or E<2 or custody['swap_count']!=10*E or custody['rejection_cap']!=100*custody['swap_count']:
        raise ValueError('Ten accepted swaps per original nonself edge and frozen100-proposal multiplier required')
    if custody.get('accepted_swaps')!=custody['swap_count'] or custody.get('proposals',0)>custody['rejection_cap']:
        raise ValueError('Complete fixed graph generation required')
    if custody.get('auxiliary_bundle_sha256')!=spec['auxiliary_graph']['sha256'] or custody.get('generator_source_sha256')!=recipe['generator_source']['sha256']:
        raise ValueError('Exact generator/output bindings required')
    if custody.get('original_edge_fingerprint')!=bindings['original_edge_fingerprint'] or custody.get('train_bundle_sha256')!=bindings['TRAIN_bundle_server_only']['sha256']:
        raise ValueError('Rewire must use the exact full factual graph, without changed features/roles')
    for key in ('exact_degree_sequence_preserved','exact_self_edges_preserved','one_common_graph_for_all_optimizer_seeds','overlap_is_construction_metadata_not_selection'):
        if custody.get(key) is not True:raise ValueError('Complete graph invariant absent: '+key)
    if custody.get('labels_used') is not False or custody.get('nodes')!=19717 or custody.get('ordered_edges')!=88648:
        raise ValueError('Complete representative public graph only, without labels')
    bind(recipe['generator_source'])
    if recipe['generator_source']['sha256']!=bindings['rewire_generator_sha256']:
        raise ValueError('Exact sealed rewire generator required')
    return custody
