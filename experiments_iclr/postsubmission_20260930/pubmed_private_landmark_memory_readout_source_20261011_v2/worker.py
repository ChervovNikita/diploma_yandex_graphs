"""Route and admission precede metadata/numerical reader work."""
import argparse
import json
from pathlib import Path
from runtime import admit, receipt, route

def main():
    route()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config['_config_receipt']=receipt(args.config)
    custody = admit(config, config['stage'])
    import reader
    result = reader.run(config, custody)
    print(json.dumps(dict(stage=config['stage'], complete=result.get('complete'),
        output=config['output'])), flush=True)

if __name__ == '__main__':
    main()
