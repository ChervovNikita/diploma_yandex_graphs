#!/usr/bin/env python3
"""Unchanged predecessor scalar,exact-state and alias diagnostic helpers."""
import argparse
import ast
import math
import json
from pathlib import Path
import time
from common import (HERE, PHASE, EXECUTION, require, sha, write as strict_write, utc, gate,
                    runtime, load_module, Progress, monitor, inventory)


def finite_json(value):
    """Preserve nonfinite telemetry as explicit tagged metadata, not invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'finite': False, 'repr': repr(value)}
    if isinstance(value, dict):
        return {key: finite_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(child) for child in value]
    return value


def write(path, value):
    strict_write(path, finite_json(value))


def exact(left, right, torch, label):
    require(type(left) is type(right), label + ': type differs')
    if isinstance(left, torch.Tensor):
        require(left.shape == right.shape and left.dtype == right.dtype and left.layout == right.layout
                and torch.equal(left, right), label + ': tensor differs')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), label + ': keys differ')
        for key in left:
            exact(left[key], right[key], torch, label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), label + ': length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, torch, label + '/' + str(index))
    else:
        require(left == right, label + ': value differs')


def differences(candidate, reference, torch, atol, rtol):
    """Scalar diagnostics for every block/tensor; reference remains on the right."""
    report = {'exact': True, 'within_fixed_numeric_rule': True, 'tensor_count': 0,
              'element_count': 0, 'maximum_absolute_difference': 0.0,
              'maximum_fraction_of_allowed_error': 0.0, 'items': []}
    def visit(a, b, name):
        row = {'path': name, 'exact': True, 'within_fixed_numeric_rule': True}
        if type(a) is not type(b):
            row.update(exact=False, within_fixed_numeric_rule=False, issue='type differs')
        elif isinstance(a, torch.Tensor):
            report['tensor_count'] += 1
            report['element_count'] += a.numel()
            row.update(shape=list(a.shape), dtype=str(a.dtype), layout=str(a.layout))
            if a.shape != b.shape or a.dtype != b.dtype or a.layout != b.layout:
                row.update(exact=False, within_fixed_numeric_rule=False, issue='tensor schema differs')
            else:
                row['exact'] = bool(torch.equal(a, b))
                if a.is_floating_point():
                    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                    row['finite'] = finite
                    if finite and a.numel():
                        delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
                        allowed = atol + rtol * b.to(torch.float64).abs()
                        row.update(maximum_absolute_difference=float(delta.max()),
                                   maximum_fraction_of_allowed_error=float((delta / allowed.clamp_min(1e-300)).max()),
                                   elements_outside_fixed_numeric_rule=int((delta > allowed).sum()),
                                   within_fixed_numeric_rule=bool((delta <= allowed).all()))
                        report['maximum_absolute_difference'] = max(report['maximum_absolute_difference'], row['maximum_absolute_difference'])
                        report['maximum_fraction_of_allowed_error'] = max(report['maximum_fraction_of_allowed_error'], row['maximum_fraction_of_allowed_error'])
                    elif not finite:
                        row.update(within_fixed_numeric_rule=False, issue='nonfinite tensor')
                else:
                    row.update(within_fixed_numeric_rule=row['exact'], unequal_elements=int((a != b).sum()))
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                row.update(exact=False, within_fixed_numeric_rule=False, issue='keys differ')
            else:
                for key in a:
                    visit(a[key], b[key], name + '/' + str(key))
                return
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                row.update(exact=False, within_fixed_numeric_rule=False, issue='length differs')
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, name + '/' + str(index))
                return
        else:
            row.update(exact=a == b, within_fixed_numeric_rule=a == b)
        report['exact'] = report['exact'] and row['exact']
        report['within_fixed_numeric_rule'] = report['within_fixed_numeric_rule'] and row['within_fixed_numeric_rule']
        report['items'].append(row)
    visit(candidate, reference, '')
    return report


def step_aliases(unit, saved, torch):
    """Only CPU Adam step scalars and storage addresses are exported as values."""
    live = unit[2].state_dict()['state']
    source = saved['Adam']['state']
    require(live.keys() == source.keys(), 'Live/saved Adam parameter IDs differ')
    rows = []
    for key in source:
        if 'step' not in source[key]:
            continue
        a, b = live[key]['step'], source[key]['step']
        require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.numel() == b.numel() == 1,
                'Qualified Adam step scalar schema differs')
        require(a.device.type == b.device.type == 'cpu', 'Non-capturable native Adam CPU steps required')
        rows.append({'parameter_id': key, 'live_step': float(a.item()), 'saved_step': float(b.item()),
                     'live_device': str(a.device), 'saved_device': str(b.device),
                     'live_data_ptr': a.data_ptr(), 'saved_data_ptr': b.data_ptr(),
                     'same_tensor_object': a is b, 'same_storage_pointer': a.data_ptr() == b.data_ptr()})
    require(rows, 'No owned Adam step scalars found')
    return rows



def tensor_bytes(value, torch):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(child, torch) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(child, torch) for child in value)
    return 0
