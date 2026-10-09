"""Prospective disjoint seed/GPU queues and metadata-only complete36 union check."""
OPERATORS=('native_tied','active_reversible_exp','pre_sigmoid_split','full_qk')
KINDS=('single','be_init','independent4')


def require(value,message):
    if not value: raise ValueError(message)


def freeze_queues(seeds, route_ids):
    require(len(seeds)==len(set(seeds))==3 and len(route_ids)==len(set(route_ids))==3
            and all(type(s)is int and 0<=s<2**63-1000000 for s in seeds),'Exactly3 fixed paired seed/route slots')
    queues={route:[dict(seed=s,operator=o,kind=k,key='seed'+str(s)+'/'+k+'/'+o)
                   for o in OPERATORS for k in KINDS] for s,route in zip(seeds,route_ids)}
    union=[cell['key'] for cells in queues.values() for cell in cells]
    require(len(union)==len(set(union))==36 and all(len(cells)==12 for cells in queues.values()),
            'Disjoint queues, full36 union, same route for every paired seed condition')
    return queues


def verify_union(*,plan,plan_sha256,receipts):
    """Trusted authenticated work/custody receipts only; no numerical result reads."""
    expected=freeze_queues(plan['seeds'],plan['route_ids'])
    require(plan['enabled']is True and plan['required_group_endpoints']==36
            and set(receipts)==set(expected) and len(plan_sha256)==64
            and all(c in '0123456789abcdef' for c in plan_sha256),'One immutable prospective complete36 plan')
    observed={}
    for route,cells in expected.items():
        receipt=receipts[route]
        require(receipt['complete']is True and receipt['clean_owned_terminal']is True
                and receipt['actual_owned_absence_attested']is True
                and receipt['plan_sha256']==plan_sha256
                and receipt['execution_source_commit']==plan['execution_source_commit']
                and receipt['route_adapter_manifest_sha256']==plan['route_adapter_manifest_sha256']
                and receipt['scores_read']is False,'Exact clean authenticated route closure before metrics')
        records=receipt['cells']
        require(set(records)=={c['key'] for c in cells},'Every assigned cell; no survivors/duplicates/reassignment')
        for spec in cells:
            record=records[spec['key']]
            require(record['cell']==spec and record['complete']is True and record['steps']==1100
                    and record['local_epochs']==100 and record['role_hashes']==plan['role_hashes']
                    and record['route_id']==route and record['physical_GPU_uuid']==plan['GPU_assignment'][route],
                    'Complete actual source/role/route cell work')
            require(spec['key']not in observed,'No duplicate scientific cell')
            observed[spec['key']]=record
    require(len(observed)==36,'All36 clean completions before any family quality opening')
    return dict(all36_complete=True,required_group_endpoints=36,route_ids=plan['route_ids'],
                metadata_only=True,comparative_opening_authorized=False,
                actual_route_custody_recheck_required_before_readout=True)
