"""Stateful longer-track selector; strict replicas delegate their native selector.

A source adapter supplies complete validation metrics and source phase gates.
No optimizer, scheduler, model or learning objective is created here.
"""
import math
class LongerTrackSelector:
 def __init__(self,selection,stopping):
  self.selection=selection;self.stopping=stopping;self.best=None;self.best_step=None;self.stale=0;self.last_step=-1
  if not selection['components']:raise ValueError('Ordered source metric/tie components required')
 def observe(self,unit,metrics,gates):
  if unit<=self.last_step:raise ValueError('Repeated/out-of-order validation')
  self.last_step=unit;key=[]
  for c in self.selection['components']:
   v=float(metrics[c['metric']])
   if not math.isfinite(v):raise ValueError('Nonfinite selection metric')
   if c['direction'] not in ('max','min'):raise ValueError('Unknown source direction')
   key.append(v if c['direction']=='max' else -v)
  key=tuple(key);eligible=unit>0 or self.selection['checkpoint0']
  better=eligible and (self.best is None or key>self.best or key==self.best and self.selection['ties']=='last')
  gate_ready=unit>=self.stopping['minimum_units'] and all(gates.get(g) is True for g in self.stopping['required_gates'])
  if better:self.best,self.best_step,self.stale=key,unit,0
  elif gate_ready:self.stale+=1
  else:self.stale=0 # pre-warmup/transition misses cannot immediately stop a new phase
  return {'save_selected':better,'stop_requested':gate_ready and self.stale>=self.stopping['patience_validation_checks'],'best_saved_step':self.best_step,'stale_validation_checks':self.stale,'required_phase_gates_complete':gate_ready}
 def capture(self):return {'selection':self.selection,'stopping':self.stopping,'best':self.best,'best_step':self.best_step,'stale':self.stale,'last_step':self.last_step}
 def restore(self,state):
  if state['selection']!=self.selection or state['stopping']!=self.stopping:raise ValueError('Changed longer-track contract')
  self.best=tuple(state['best']) if state['best'] is not None else None;self.best_step=state['best_step'];self.stale=state['stale'];self.last_step=state['last_step']
