"""Typed exact publicV2 Session and coherent detached old-view-B corrector source law."""
from adapters.label_corrector import CONTRACT,WORK
def validate_label_corrector_run(run):
 c,b,s=run['config'],run['budget'],run['selection'];arm=c.get('label_corrector_arm');w=WORK.get(arm);seed=run['seed'];body=run['body']
 if w is None:raise ValueError('Literal source labelcorrector arm required')
 members,heads,back,adam=w;original=arm=='original_GNNM';native_forwards=8 if original else 2;native_backwards=8 if original else 1
 expected={'stage':'comparative','label_corrector_source_contract':CONTRACT,'backbone':'literal_publicV2_WikiCS_Session_Polynormer','members':members,'physical_slow_bodies':1,'source_optimizer_count':1+adam,'source_complete_body_constructors':1,'source_corrector_head_constructors':1 if arm=='label_only4_detached' else heads,'source_native_TRAIN_forwards_per_epoch':native_forwards,'source_native_TRAIN_backwards_per_epoch':native_backwards,'source_corrector_TRAIN_head_forwards_per_epoch':heads,'source_corrector_TRAIN_backwards_per_epoch':back,'source_corrector_Adam_maps_per_epoch':adam,'source_common_query_masks_per_epoch':int(heads>0),'width':512,'depth':7,'global_layers':2,'classes':10,'learning_rate':.001,'weight_decay':0.,'local_transition_epochs':100,'initialization_seed':seed+1009*body,'is_original_gnnm':original,'historical_state_inherited':False,'fusion':'mean_logits' if original else 'mean_probabilities'}
 if (seed not in (6101,6203,6307) or body not in ((0,1,2,3) if arm=='native_single' else (0,)) or any(c.get(k)!=v for k,v in expected.items())
     or b!={'unit':'epochs','policy':'full_budget','maximum':1100}
     or s!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}
     or not run['adapter'].endswith('/adapters/label_corrector.py:make_adapter') or run['parents']):
  raise ValueError('Exact Session1100epochs/local100,detachedOLDviewB,coherentownlocalrestore/liveendRNG andfixedsourcecorrector owners required')
 return {'optimizer_objects':1+adam,'physical_slow_bodies':1,'known_optimizer_step_calls':1100*(1+adam),'budget_unit':'epochs','registered_budget_units':1100,'law':'literal_PublicV2_Session_labelcorrector_oldviewB_1100_coherentlocal100',
  'known_source_TRAIN_forward_calls':1100*(native_forwards+heads),'known_source_TRAIN_backward_calls':1100*(native_backwards+back),
  'known_label_corrector_native_TRAIN_forwards':1100*native_forwards,'known_label_corrector_native_TRAIN_backwards':1100*native_backwards,'known_label_corrector_TRAIN_head_forwards':1100*heads,'known_label_corrector_TRAIN_backwards':1100*back,'known_label_corrector_Adam_maps':1100*adam,'known_label_corrector_native_Adam_maps':1100,'known_label_corrector_common_masks':1100*int(heads>0),'known_label_corrector_coherent_local_restores':1,'unstepped_pre_install_Adam_constructor_objects':int(original)}
