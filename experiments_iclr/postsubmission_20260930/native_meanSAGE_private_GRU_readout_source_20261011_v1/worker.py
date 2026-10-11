"""Route and admission precede metadata/numerical reader work."""
import argparse
import json
from pathlib import Path
from runtime import admit, route

def main():
    route()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    custody = admit(config, config['stage'])
    import reader
    result = reader.run(config, custody)
    print(json.dumps(dict(stage=config['stage'], complete=result.get('complete'),
        output=config['output'])), flush=True)

if __name__ == '__main__':
    main()
