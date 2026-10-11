"""Reconstruct the exact sealed complete summary from all small partitions."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    index=json.loads((here/'INDEX.json').read_text())
    values={}
    for row in index['parts']:
        raw=(here/row['name']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        values[row['name']]=json.loads(raw)
    value=values['METADATA.json']
    value['records']=values['ALL30_BANK_RECORDS.json']
    value['comparisons']=sum([values[row['name']] for row in index['parts'] if row['name'].startswith('ORDERED_COMPARISONS_')],[])
    assert len(value['records'])==30 and len(value['comparisons'])==90
    value={key:value[key] for key in index['source_key_order']}
    data=(json.dumps(value,indent=2)+'\n').encode()
    assert len(data)==index['source_summary']['bytes'] and hashlib.sha256(data).hexdigest()==index['source_summary']['sha256']
    with args.output.open('xb') as output:output.write(data)
    print(json.dumps(dict(output=str(args.output.resolve()),bytes=len(data),sha256=index['source_summary']['sha256'],complete_numeric_summary=True)))

if __name__=='__main__':main()
