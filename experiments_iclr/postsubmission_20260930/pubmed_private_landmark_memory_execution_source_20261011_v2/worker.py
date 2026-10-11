"""Literal route and finite root stage admission before numerical worker imports."""
import argparse
import importlib
import json
from pathlib import Path
from runtime import admit,receipt,route

def main():
    route();parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);args=parser.parse_args()
    config=json.loads(args.config.read_text());custody=admit(config,config['stage']);config['_config_receipt']=receipt(args.config)
    result=importlib.import_module(config['stage']).run(config,custody)
    print(json.dumps(dict(stage=config['stage'],complete=result['complete'],output=config['output'])),flush=True)

if __name__=='__main__':main()
