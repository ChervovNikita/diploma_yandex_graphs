"""Source and mocked factory contracts only; no numerical/native construction."""
import ast,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
E=Path(__file__).resolve().parents[2]
PAIRS=[('link_expansion_ports_BUDDY_v2','link_expansion_ports_BUDDY_production_v1','BUDDYExpandedSession'),('buddy_private_bn_state_v1','buddy_private_bn_production_v1','BUDDYPrivateBN'),('interaction_BUDDY_v1','interaction_BUDDY_production_v2','BUDDYInteractionSession')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
def load(p):
    spec=importlib.util.spec_from_file_location('_contract_'+sha(p),p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def main():
    report={'schema':'BUDDY-production-successor-zero-training-contracts-v2','status':'running','packages':[],'mock_factory_calls':0,'rejected_invalid_contracts':0,'native_constructions':0,'native_forwards':0,'optimizer_updates':0,'sampler_calls':0,'TEST_accessed':False}
    for oldname,newname,cls in PAIRS:
        old,new=E/oldname,E/newname
        before,after=functions(old/'session.py'),functions(new/'session.py')
        preserved=['assert_identity','train_update','train_epoch','metadata','close']
        if cls=='BUDDYExpandedSession':preserved+=['_finite_gradients','_step','observe_model','next_batch','step','state','load_state']
        elif cls=='BUDDYPrivateBN':preserved+=['_snapshot','_counts','_audit_TRAIN','require_route','require_BN','state_export','before_load','route','next_batch']
        else:preserved+=['_finite_gradients','_recipe_receipt','_step','selected_mean_sidecar','forward','next_batch','step','state']
        assert all(before[name]==after[name] for name in preserved),[(name,before[name]==after[name])for name in preserved]
        if cls=='BUDDYInteractionSession':
            assert (old/'objectives.py').read_bytes()==(new/'objectives.py').read_bytes()
            old_load=(old/'session.py').read_text();new_load=(new/'session.py').read_text()
            expected=before['load_state'].replace("Constant(value='native')", "Constant(value='native')")
            # The sole restore-law delta is the explicitly added provider checkpoint layer.
            repaired=old_load.replace("native_state = value['native_integration']['native']","native_state = value['native_integration']['native_integration']['native']")
            expected={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(repaired))if isinstance(n,ast.FunctionDef)}['load_state']
            assert expected==after['load_state']
        source=(new/'session.py').read_text()
        assert 'native_context(' not in source and "['TRAIN_batches']" not in source and "['VALID_inputs']" not in source
        for p in new.glob('*.py'):
            tree=ast.parse(p.read_text())
            calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
            assert not any(name.endswith('manual_seed') or name in ('random.seed','np.random.seed','torch.load','torch.save')for name in calls)
        cb=load(new/'callback.py');library=cb.source();calls=[]
        methods=list(library.METHODS) if hasattr(library,'METHODS') else [library.METHOD]
        original_bindings,original_load,original_class=library.bindings,library.load,getattr(library,cls)
        class Outer:
            def __init__(self,base,job,pins,*args):self.base=base;self.job=job;self.model=base.model;self.optimizers=base.optimizers
        def factory(torch,job,policy,root):
            calls.append(copy.deepcopy(job));base=types.SimpleNamespace(model=object(),optimizers=[object()])
            return types.SimpleNamespace(session=base,model=base.model,optimizers=base.optimizers)
        def provider_load(path,*args,**kwargs):
            assert Path(path)==E/'buddy_native_provider_v1/callback.py';return types.SimpleNamespace(create_adapter=factory)
        library.load=provider_load;library.bindings=lambda *args: (['source-pin'],{'source':'declared'}) if cls=='BUDDYInteractionSession' else ['source-pin'];setattr(library,cls,Outer)
        try:
            for method in methods:
                for seed in (0,1,9):
                    job={'family':'link','procedure':method,'base_procedure':'gnnm_boundary_native','TEST_access':False,'seed':seed,'recipe':{'architecture':'BUDDY','_repository_root':str(E.parent.parent),'_seed':seed,'seed':seed},'callback_module':str(new/'callback.py'),'callback_sha256':sha(new/'callback.py'),'method_settings':{'manifest_sha256':'method-pin','provider_manifest_sha256':'provider-pin','static_provider_sha256':'static-pin','provider_mode':'production_native'}}
                    if cls=='BUDDYPrivateBN':job['method_settings']['banks_sha256']='bank-pin'
                    original=copy.deepcopy(job);adapter=cb.create_adapter(object(),job,types.SimpleNamespace(family='link',track='matched_extended'),E.parent.parent)
                    assert job==original and adapter.session.job==original
                    inner=calls[-1];assert inner['seed']==seed and inner['recipe']==job['recipe'] and inner['procedure']=='gnnm_boundary_native'
                    assert inner['method_settings']=={'manifest_sha256':'provider-pin','static_provider_sha256':'static-pin','provider_mode':'production_native'}
                    assert inner['callback_module']==str(E/'buddy_native_provider_v1/callback.py') and inner['callback_sha256']==sha(E/'buddy_native_provider_v1/callback.py')
                    report['mock_factory_calls']+=1
            bad_jobs=[]
            for mutation in ('oracle_field','missing_provider','wrong_mode','negative_seed','bool_seed','wrong_callback','TEST_open'):
                bad=copy.deepcopy(job)
                if mutation=='oracle_field':bad['method_settings']['context_manifest']='forbidden'
                elif mutation=='missing_provider':bad['method_settings'].pop('provider_manifest_sha256')
                elif mutation=='wrong_mode':bad['method_settings']['provider_mode']='smoke_oracle'
                elif mutation=='negative_seed':bad['seed']=-1
                elif mutation=='bool_seed':bad['seed']=True
                elif mutation=='wrong_callback':bad['callback_sha256']='wrong'
                else:bad['TEST_access']=True
                count=len(calls)
                try:cb.create_adapter(object(),bad,types.SimpleNamespace(family='link',track='matched_extended'),E.parent.parent)
                except ValueError:pass
                else:raise AssertionError('Invalid callback contract admitted: '+mutation)
                assert len(calls)==count;report['rejected_invalid_contracts']+=1
        finally:library.bindings,library.load=original_bindings,original_load;setattr(library,cls,original_class)
        report['packages'].append({'package':newname,'parent_manifest_sha256':sha(old/'MANIFEST.json'),'AST_identical_method_functions':preserved,'provider_factory_only':True,'no_oracle_read_or_seed_reset':True,'callback_sha256':sha(new/'callback.py'),'session_sha256':sha(new/'session.py')})
    # The retained resolver deliberately unwraps only a LinkSession with own native_binding/native.
    # Resolve the explicit actual native object across the added provider layer.
    mapping=load(E/'portable_private_input_v2/extension.py')
    def affine(shape):return types.SimpleNamespace(weight=types.SimpleNamespace(shape=shape,ndim=2))
    model=types.SimpleNamespace(label_lin_layer=affine((16,16)),lin_feat=affine((1024,128)),lin=affine((1,1040)),sign_k=0)
    core=types.SimpleNamespace(banks=object(),binding_metadata=[{'site':'label_lin_layer','first_input':True},{'site':'lin_feat','first_input':True},{'site':'lin','first_input':False}])
    native=types.SimpleNamespace(core=core,model=model,members=4,procedure='base',metadata={'family':'BUDDY'})
    class ProviderProxy:
        def __init__(self,n):self.native=n
        def __getattr__(self,name):return getattr(self.native,name)
        def metadata(self):return {'provider':'production_native'}
    proxy=ProviderProxy(native)
    try:mapping.source_mapping(proxy,'link')
    except TypeError:pass
    else:raise AssertionError('Resolver contract changed; old excluded proxy route unexpectedly admitted')
    owner,m,c,inputs,output=mapping.source_mapping(proxy.native,'link')
    assert owner is native and m is model and c is core and [(x[0],x[2])for x in inputs]==[('label_lin_layer',16),('lin_feat',1024)] and output[0]=='lin'
    tree=ast.parse((E/'interaction_BUDDY_production_v2/session.py').read_text())
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='mapping.source_mapping']
    assert len(calls)==1 and ast.unparse(calls[0].args[0])=='n'
    report['native_mapping_contract']={'old_proxy_dispatch_rejected':True,'explicit_owned_native_mapping_passed':True,'backbone_constructions':0,'numerical_tensors':0}
    assert 'torch' not in sys.modules,'This source-only check unexpectedly imported torch'
    report['status']='passed';out=Path(__file__).parent/'RESULT.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
