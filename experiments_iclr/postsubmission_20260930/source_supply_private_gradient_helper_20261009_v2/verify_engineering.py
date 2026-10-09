"""Fixed stdlib-only projection checks and one synthetic full-dimension benchmark."""
import ast
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREDECESSOR = HERE.parent/'source_supply_private_gradient_helper_20261009_v1'/'source_supply.py'
TOL = 1e-9
DIMENSION = 3 * 97536


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walsh(code,index):
    return -1 if bin(code & index).count('1') % 2 else 1


def engineering_vectors(dimension):
    assert dimension % 16 == 0
    patterns=tuple(tuple(walsh(code,j) for j in range(16)) for code in range(11))
    rows=tuple(tuple(float(pattern[j % 16]) for j in range(dimension)) for pattern in patterns)
    numerator=[4*sum(walsh(code,j) for code in range(11))+walsh(15,j) for j in range(16)]
    target=tuple(float(numerator[j % 16])/4 for j in range(dimension))
    return target,rows


def usage():
    value=resource.getrusage(resource.RUSAGE_SELF)
    return dict(user=value.ru_utime,system=value.ru_stime,
                cumulative_RSS_peak_bytes=int(value.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))


def scope(start,before):
    after=usage()
    return dict(wall_seconds=time.perf_counter()-start,CPU_user_seconds=after['user']-before['user'],
                CPU_system_seconds=after['system']-before['system'],cumulative_RSS_peak_bytes=after['cumulative_RSS_peak_bytes'])


def main():
    total_start,total_usage=time.perf_counter(),usage()
    old=load('frozen_projection_reference',PREDECESSOR)
    new=load('cached_Gram_projection_successor',HERE/'source_supply.py')
    assert sha(PREDECESSOR)=='b8bcbbede340b7850a09bbe01cd1c62d013d569abda34116c34cefdac22382e9'
    old_tree,new_tree=ast.parse(PREDECESSOR.read_text()),ast.parse((HERE/'source_supply.py').read_text())
    assert len(old_tree.body)==len(new_tree.body)
    for a,b in zip(old_tree.body,new_tree.body):
        if isinstance(a,ast.FunctionDef) and a.name=='project_nonincrease_cone':
            continue
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    try:
        new.Config().require_enabled()
    except new.ContractError:
        pass
    else:
        raise AssertionError('Default helper must remain disabled')
    cases=[
        ('no_constraints',(1.0,-2.0),()),
        ('zero_row',(1.0,-2.0),((0.0,0.0),)),
        ('orthant',(1.0,-2.0,3.0),((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))),
        ('oblique_inactive',(2.0,1.0,-0.5),((1.0,1.0,0.0),(0.0,1.0,1.0))),
        ('redundant_opposing',(2.0,-1.0),((1.0,0.0),(2.0,0.0),(-1.0,0.0))),
    ]
    target,rows=engineering_vectors(16)
    cases.append(('all11_dense16',target,rows))
    small_start,small_usage=time.perf_counter(),usage()
    comparisons=[]
    for name,target,rows in cases:
        original=old.project_nonincrease_cone(target,rows,TOL)
        successor=new.project_nonincrease_cone(target,rows,TOL)
        difference=max(abs(a-b) for a,b in zip(original,successor))
        assert difference <= TOL
        comparisons.append(dict(name=name,dimension=len(target),risk_rows=len(rows),
            max_abs_coordinate_difference=difference,comparison_bound=TOL))
    rejection=[]
    for name,target,rows in [('nonfinite_target',(float('nan'),),()),
                             ('nonfinite_risk',(1.0,),((float('inf'),),))]:
        for module in (old,new):
            try:
                module.project_nonincrease_cone(target,rows,TOL)
            except module.ContractError:
                pass
            else:
                raise AssertionError('Finite guard must reject')
        rejection.append(name)
    small_cost=scope(small_start,small_usage)
    del cases,target,rows
    gc.collect()

    benchmark_start,benchmark_usage=time.perf_counter(),usage()
    generation_start,generation_usage=time.perf_counter(),usage()
    target,rows=engineering_vectors(DIMENSION)
    generation_cost=scope(generation_start,generation_usage)
    original_dot=new._dot
    counts=Counter()
    def observed_dot(a,b):
        counts[len(a)]+=1
        return original_dot(a,b)
    new._dot=observed_dot
    projection_start,projection_usage=time.perf_counter(),usage()
    try:
        projected=new.project_nonincrease_cone(target,rows,TOL)
    finally:
        new._dot=original_dot
    projection_cost=scope(projection_start,projection_usage)
    expected_difference=max(abs(value-walsh(15,j % 16)/4) for j,value in enumerate(projected))
    assert expected_difference <= TOL
    assert len(projected)==DIMENSION and all(math.isfinite(value) for value in projected)
    assert counts[DIMENSION]==89  # 66 Gram,11 target,11 final feasibility,1 final distance.
    benchmark_cost=scope(benchmark_start,benchmark_usage)
    assert not {'torch','numpy','dgl','torch_sparse','sklearn'}.intersection(sys.modules)
    receipt=dict(schema='cached-Gram-fixed-stdlib-engineering-verification-v2',status='passed',
        runtime=dict(python_version=platform.python_version(),python_executable=sys.executable,platform=platform.platform()),
        source_sha256=sha(HERE/'source_supply.py'),predecessor_source_sha256=sha(PREDECESSOR),
        only_changed_module_AST_node='project_nonincrease_cone',original_tolerance=TOL,
        small_equivalence_cases=comparisons,finite_guard_cases=rejection,small_checks_cost=small_cost,
        representative_benchmark=dict(dimension=DIMENSION,risk_rows=11,active_subsets_enumerated=2048,
            vector_family='deterministic16-period Walsh rows0-10 with orthogonal Walsh15 target component1/4',
            expected_solution_max_abs_coordinate_error=expected_difference,validation_bound=TOL,
            high_dimension_dot_calls=counts[DIMENSION],dot_calls_by_dimension=dict(sorted(counts.items())),
            generation_cost=generation_cost,projection_cost=projection_cost,total_cost=benchmark_cost,
            predecessor_high_dimension_benchmark_run=False),
        total_observed_cost=scope(total_start,total_usage),
        limits=['Synthetic engineering vectors only; no model gradients, labels, datasets or scientific effects.',
                'One benchmark execution and six fixed small cases; no float tolerance search or performance extrapolation.',
                'Original and compressed formulas are equal in real arithmetic, not promised bitwise equal in finite arithmetic.',
                'The reported RSS peak is cumulative for this local process, not isolated model or projection incremental memory.',
                'This source does not qualify native V2 bank/qualifier successor integration.'],
        default_disabled=True,numerical_provider_model_data_checkpoint_outcome_server_execution=False)
    (HERE/'ENGINEERING_VERIFICATION.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='passed',source_sha256=receipt['source_sha256'],
        projection_cost=projection_cost,total_observed_cost=receipt['total_observed_cost'],
        representative_max_abs_error=expected_difference,high_dimension_dot_calls=counts[DIMENSION])))


if __name__=='__main__':
    main()
