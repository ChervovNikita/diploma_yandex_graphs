"""Render only the reviewed complete18 release and required-reference declaration."""
import argparse
import json
from pathlib import Path
from ENTRY import HERE,route,frozen,approvals,admitted,descriptor


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    route();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-review',type=Path,required=True);p.add_argument('--owner-review',type=Path,required=True)
    args=p.parse_args();b,manifest_sha=frozen();root_review=descriptor(args.root_review);owner_review=descriptor(args.owner_review)
    approvals(b,manifest_sha,root_review,owner_review)
    policy=json.loads((HERE/'INDEPENDENT_REFERENCE_PROTOCOL_TEMPLATE_DISABLED.json').read_text());policy.update(enabled=True,root_source_review_approved=True)
    policy_path=HERE/'INDEPENDENT_REFERENCE_PROTOCOL.json';write(policy_path,policy)
    spec=json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
    for k in ('enabled','root_source_review_approved','data_scope_approved','provider_runtime_approved','scientific_execution_approved'):spec[k]=True
    spec.update(capabilities={k:True for k in ('source_bound','model','data','runtime','scientific')},activation_source_manifest_sha256=manifest_sha,
                root_review=root_review,owner_review=owner_review,independent_reference_protocol=descriptor(policy_path))
    admitted(spec,b,manifest_sha);path=HERE/'RELEASE.json';write(path,spec)
    print(json.dumps(dict(release=descriptor(path),required_reference_protocol=descriptor(policy_path),all18=True,nothing_started=True)))


if __name__=='__main__':main()
