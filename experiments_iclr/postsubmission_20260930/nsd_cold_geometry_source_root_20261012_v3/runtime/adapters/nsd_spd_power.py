"""Same native SVD forward;continuous SPD inverse-square-root first adjoint only."""
import types


def make_spd_inverse_sqrt(torch):
 class SourceForwardStableAdjoint(torch.autograd.Function):
  @staticmethod
  def forward(ctx,matrices,p):
   if p != -.5 or matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:raise ValueError('Only the declared augmented SPD inverse-square-root block operator')
   if not bool(torch.isfinite(matrices).all()) or not torch.equal(matrices,matrices.transpose(-1,-2)):raise ValueError('Exact finite symmetric source Gram blocks required')
   if bool((torch.linalg.cholesky_ex(matrices)[1]!=0).any()):raise ValueError('Strictly positive definite source blocks required')
   # Identical source operations/order.Original source SVD/forward/cutoff retained.
   vecs,vals,_=torch.linalg.svd(matrices)
   good=vals>vals.max(-1,True).values*vals.size(-1)*torch.finfo(vals.dtype).eps
   if not bool(good.all()) or not bool((vals>0).all()):raise ValueError('Prospective SPD derivative does not replace source-masked/nonpositive spectra')
   powered=vals.pow(p).where(good,torch.zeros((),device=matrices.device,dtype=matrices.dtype))
   matrix_power=(vecs*powered.unsqueeze(-2))@torch.transpose(vecs,-2,-1)
   ctx.save_for_backward(vecs,vals)
   return matrix_power
  @staticmethod
  @torch.autograd.function.once_differentiable
  def backward(ctx,gradient):
   vecs,vals=ctx.saved_tensors;roots=vals.sqrt();left=roots.unsqueeze(-1);right=roots.unsqueeze(-2)
   divided=-1/(left*right*(left+right))
   symmetric=(gradient+gradient.transpose(-1,-2))*.5
   coordinates=vecs.transpose(-1,-2)@symmetric@vecs
   result=vecs@(divided*coordinates)@vecs.transpose(-1,-2)
   return result,None
 return SourceForwardStableAdjoint.apply


def bind_constant_incidence_adjoint(torch,model):
 """Bind exact original normalise code with only its matrix-power lookup replaced."""
 power=make_spd_inverse_sqrt(torch)
 class LapProxy:
  def __init__(self,base):self.base=base
  def __getattr__(self,name):return power if name=='batched_sym_matrix_pow' else getattr(self.base,name)
 count=0
 for member in model.members:
  builder=member.laplacian_builder
  if type(builder).__name__!='GeneralLaplacianBuilder' or not builder.normalised or not builder.augmented:raise ValueError('Exact full-General augmented source normalization required')
  if any(type(learner).__name__!='LayerSharedIncidence' for learner in member.sheaf_learners):raise ValueError('Correction is restricted to the two declared constant-map source methods')
  original=builder.normalise.__func__;namespace=dict(original.__globals__);namespace['lap']=LapProxy(namespace['lap'])
  replacement=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
  replacement.__kwdefaults__=original.__kwdefaults__;replacement.__annotations__=original.__annotations__
  builder.normalise=types.MethodType(replacement,builder);count+=1
 if count!=4:raise ValueError('Four complete private source builders required')
 return {'normalizers':count,'source_normalise_code_unchanged':True,'source_SVD_forward_unchanged':True,'source_training_jitter_and_RNG_unchanged':True,'matrix_function_first_adjoint':'continuous_SPD_inverse_sqrt_Frechet','higher_derivatives_supported':False}
