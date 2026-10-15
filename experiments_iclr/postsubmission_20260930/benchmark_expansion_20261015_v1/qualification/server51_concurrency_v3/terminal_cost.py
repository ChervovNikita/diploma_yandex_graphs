"""Conservative scalar terminal-cost completeness; never infer missing work zero."""
import math

def complete_terminal_cost(result,exit_code,*,expected,forced_KILL=False,cleanup_error=None):
 if not isinstance(result,dict) or exit_code!=0 or result.get('status')!='PASS' or forced_KILL or cleanup_error or result.get('close_error'):return False
 if any(result.get(key)!=value for key,value in expected.items()):return False
 for key in ('TRAIN_epochs_completed','VALID_calls_completed'):
  if type(result.get(key)) is not int or result[key]<0:return False
 calls=result.get('optimizer_calls')
 if not isinstance(calls,dict) or set(calls)!={'attempts','completed'}:return False
 if any(not isinstance(calls[key],list) or len(calls[key])!=1 or type(calls[key][0]) is not int or calls[key][0]<0 for key in calls):return False
 if calls['attempts'][0]<calls['completed'][0]:return False
 wall=result.get('invocation_wall_seconds')
 return type(wall) in (int,float) and math.isfinite(wall) and wall>=0
