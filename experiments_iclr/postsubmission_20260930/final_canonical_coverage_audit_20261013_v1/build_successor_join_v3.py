"""Successor exact join after source ports close; candidate24 audit stays immutable."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
sp=importlib.util.spec_from_file_location('candidate24_join_engine',H/'build_join.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rootproof='final_canonical_coverage_audit_20261013_v1/source_port_successor_metadata/ALLOCATION_HANDOFF.json';structproof='native_structural_cardinality_source_ports_20261013_v1/CANONICAL_PROCEDURE_COVERAGE.json';postproof='native_postfit_serving_ports_20261013_v2/CLOSED_HANDOFF.json'
policies=['source_copied_relation_credit_'+a for a in ('alphaF','allJ','phiJ','relationJ')]
m.MAP.update({
 'graph_relation_private_credit_configured_relation12':(policies,{'configured_relation_source_contract.schema':'copied-native-relation-credit-source1100-v1'}),
 'graph_relation_private_credit_configured_relation18':(policies+['label_only_corrector_native_single','label_only_corrector_genuine_native_I4'],{'configured_relation_source_contract.schema':'copied-native-relation-credit-source1100-v1'}),
 'private_hop_private_hop_controls':(['literal_hop_'+a for a in ('full_aux','common_nonfull','allblock_missinghop','factor1_allview')],{}),
 'structural_pattern_exact_count_conditioned':(['conditional_pattern_'+a for a in ('J_K','J_K_sep','W_K')],{}),
 'structural_pattern_exact_support_bucketed':(['conditional_pattern_'+a for a in ('J_K','J_K_sep','W_K')],{}),
 'structural_pattern_cardinality_single':(['structural_pattern_cardinality_single'],{'cardinality_arm':'D4'}),
 'structural_pattern_private_structural_interaction':(['structural_pattern_private_structural_interaction'],{'structural_arm':'structural_private'}),
 'learned_graph_serving_self_reliability':(['learned_graph_serving_self_reliability'],{'postfit_rule':'reliability_self','postfit_phase':'head'}),
 'learned_graph_serving_graph_reliability':(['learned_graph_serving_graph_reliability'],{'postfit_rule':'reliability_graph','postfit_phase':'head'}),
 'e_fixed_feature_head_capacity_affine_head':(['e_fixed_feature_head_capacity_affine_head'],{'postfit_family':'affine','postfit_phase':'head'}),
})
for name,condition in [('be_factor_refit','BE_factor_refit'),('be_full_refit','BE_full_refit'),('ordinary_full_refit','ordinary_full_refit'),('single_full_refit','single_full_refit')]:m.MAP['direct_hidden_head_refit_'+name]=(['direct_hidden_head_refit_'+name],{'postfit_condition':condition,'postfit_phase':'head'})
for key in m.RESIDUAL:
 m.PROOF_FILES[key]=rootproof if key.startswith(('graph_relation_private_credit_configured','private_hop_private_hop_controls')) else structproof if key.startswith('structural_pattern_') else postproof
# Every formerly residual row now needs a real exact match; zero matches remain missing.
m.RESIDUAL={}
if __name__=='__main__':m.main()
