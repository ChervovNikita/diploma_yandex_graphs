"""Source preparation and engineering gates; all real commands fail closed."""
import argparse
import json
import sys
sys.dont_write_bytecode=True
from contract import HERE,RECIPE,LAYOUTS,OBJECTIVES,sources,matrix,write_new,refuse_real


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','runtime-freeze','preflight128','train','replay'])
    p.add_argument('--output',required=True)
    p.add_argument('--binding')
    p.add_argument('--supervisor')
    p.add_argument('--seed',type=int,choices=[0,1,2])
    p.add_argument('--objective',choices=OBJECTIVES)
    p.add_argument('--layout',choices=LAYOUTS)
    p.add_argument('--run')
    a=p.parse_args()
    if a.command in ('preflight128','train','replay'):
        refuse_real() # Before binding, checkpoint, torch or dataset import.
    if a.command=='prepare':
        value={'schema':'gnnm-primary-molhiv-source-preparation-v1','status':'source_only_unlaunched','sources':sources(),'recipe':RECIPE,'primary_complete_matrix':matrix(),'real_admission':False,'no_test_subset_or_scoring':True}
        write_new(a.output,value);print(json.dumps({'status':value['status'],'primary_fits':30}));return
    if a.supervisor is None:raise ValueError('Runtime freeze requires live authorized supervisor')
    from predictive_core import configure,runtime_freeze
    configure();runtime_freeze(a.output,a.supervisor)


if __name__=='__main__':main()
