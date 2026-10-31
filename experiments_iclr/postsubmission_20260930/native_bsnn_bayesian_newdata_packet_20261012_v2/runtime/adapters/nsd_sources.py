"""Owned pinned full general NSD imports; isolated legacy models/lib namespaces."""
import ast,copy,hashlib,importlib,importlib.util,math,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
from .native_label_families import bound,load_module

CACHE={}


def extract(root,row,names,env):
 tree=ast.parse(bound(root,row).read_bytes());nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Pinned complete source definitions required')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),row['path']+'::exact_definitions','exec'),env);return SimpleNamespace(**env)


def native(torch,root,B):
 directory=bound(root,B['native/models/disc_models.py']).parents[1];key=str(directory.resolve())
 pins=__import__('json').loads(bound(root,B['pins']).read_bytes())
 for rel,digest in pins.items():
  if hashlib.sha256((directory/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Every original native NSD file pinned')
 if key in CACHE:return CACHE[key]
 prior={n:v for n,v in sys.modules.items() if n in ('models','lib') or n.startswith(('models.','lib.'))}
 for n in prior:sys.modules.pop(n)
 path=list(sys.path)
 try:
  sys.path.insert(0,str(directory))
  package=ModuleType('models');package.__path__=[str(directory/'models')];package.__file__=str(directory/'models/__init__.py');sys.modules['models']=package
  ortho_path=directory/'models/orthogonal.py';module=ModuleType('models.orthogonal');module.__file__=str(ortho_path)
  def unavailable(*args,**kwargs):raise RuntimeError('Orthogonal-family Householder path is outside this full-General NSD packet and unavailable')
  env=module.__dict__;env.update(math=math,torch=torch,nn=torch.nn,torch_householder_orgqr=unavailable)
  tree=ast.parse(ortho_path.read_bytes());defs=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
  exec(compile(ast.Module(body=defs,type_ignores=[]),str(ortho_path)+'::unchanged_class_without_unused_import','exec'),env);sys.modules['models.orthogonal']=module
  builders=ast.parse((directory/'models/laplacian_builders.py').read_bytes());general=next(n for n in builders.body if isinstance(n,ast.ClassDef) and n.name=='GeneralLaplacianBuilder')
  if any(isinstance(n,ast.Name) and n.id=='Orthogonal' for n in ast.walk(general)):raise ValueError('This exact full-General source must not use optional Householder')
  model=importlib.import_module('models.disc_models')
  for n,m in sys.modules.items():
   if n in ('models','lib') or n.startswith(('models.','lib.')):
    if not getattr(m,'__file__',None) or not Path(m.__file__).resolve().is_relative_to(directory):raise ValueError('Owned native namespace only')
  CACHE[key]=model.DiscreteGeneralSheafDiffusion;return CACHE[key]
 finally:
  for n in list(sys.modules):
   if n in ('models','lib') or n.startswith(('models.','lib.')):sys.modules.pop(n)
  sys.modules.update(prior);sys.path[:]=path


def environment(torch,root,B):
 for row in B.values():bound(root,row)
 adapter=load_module(root,B['adapter']);adapter.load_native_general_class=lambda:native(torch,root,B)
 placement=load_module(root,B['placement'])
 require=lambda v,msg:None if v else (_ for _ in ()).throw(ValueError(msg))
 geometry=extract(root,B['geometry_bank'],('make_bank',),{'copy':copy,'require':require})
 train=extract(root,B['geometry_train'],('train_update','optimizer'),{'require':require})
 baseline=extract(root,B['baseline'],('metrics',),{})
 init=extract(root,B['live_init'],('FirstActionCapture','_hash','_directions','_norm','_pair_choice'),{'hashlib':hashlib,'itertools':__import__('itertools'),'require':require})
 return SimpleNamespace(adapter=adapter,placement=placement,geometry=geometry,train=train,baseline=baseline,init=init,require=require)
