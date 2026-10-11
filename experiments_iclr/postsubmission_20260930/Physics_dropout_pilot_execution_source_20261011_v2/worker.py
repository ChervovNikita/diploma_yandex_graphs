"""Stage worker; route verification precedes project/config reads and numerics."""
import argparse
import importlib
import json
from pathlib import Path
from runtime import admit,route

def main():
    route();parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);args=parser.parse_args()
    config=json.loads(args.config.read_text());stage=config['stage'];custody=admit(config,stage)
    if stage not in ('custody','qualify','acquire','calibrate','score'):raise ValueError('Unknown fixed stage')
    module=importlib.import_module({'custody':'custodian','qualify':'qualify','acquire':'acquire','calibrate':'reader','score':'reader'}[stage])
    result=module.run(config,custody)
    print(json.dumps(dict(stage=stage,complete=result.get('complete'),output=config['output'])),flush=True)
if __name__=='__main__':main()
