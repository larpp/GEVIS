"""Resolve the existing TIPS profiles without inspecting held-out sensor values."""
import copy
import json
import re
from pathlib import Path

from .tips_dataset import sha


def resolve_config(config, gevis_root):
    result = copy.deepcopy(config)
    keys = {'step_profile', 'sensor_profile'}
    if not keys.intersection(config):
        return result  # Explicit legacy/test fixture input, not the default user entry.
    if not keys <= config.keys():
        raise ValueError('Both step_profile and sensor_profile are required')
    conflicting = {'main_steps', 'target_sensors', 'columns_file', 'selected_sensors'} & config.keys()
    if conflicting:
        raise ValueError('Profiles are the sole selection authority; remove overrides: ' + str(sorted(conflicting)))
    root = Path(gevis_root).resolve()
    paths = {name: (root / config[name]).resolve() for name in keys}
    paths['sensor_dictionary'] = (root / config.get('sensor_dictionary',
        str(Path(config['tips_root']) / 'config/sensor_dictionary.json'))).resolve()
    snapshots = {name: json.loads(path.read_text(encoding='utf-8-sig')) for name, path in paths.items()}
    step = snapshots['step_profile']; sensor = snapshots['sensor_profile']
    if set(step) != {'schema_version', 'include', 'exclude'} or type(step['schema_version']) is not int or step['schema_version'] != 1:
        raise ValueError('Step profile requires schema_version=1, include and exclude')
    for key in ['include', 'exclude']:
        values = step[key]
        if (not isinstance(values, list) or any(type(v) is not int or not 0 <= v < 255 for v in values)
                or len(values) != len(set(values))):
            raise ValueError('Step include/exclude requires unique integers 0..254')
    steps = sorted(set(step['include']) - set(step['exclude']))
    if not steps:
        raise ValueError('Step profile selects no Main Steps')
    if set(sensor) != {'schema_version', 'groups', 'include', 'exclude'} or type(sensor['schema_version']) is not int or sensor['schema_version'] != 1:
        raise ValueError('Sensor profile requires schema_version=1, groups, include and exclude')
    for key in ['groups', 'include', 'exclude']:
        if not isinstance(sensor[key], list) or any(not isinstance(v, str) or not v.strip() for v in sensor[key]):
            raise ValueError('Sensor groups/include/exclude must be arrays of nonempty strings')
    dictionary = snapshots['sensor_dictionary']
    names = [entry['name'].strip() for entry in dictionary]
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous dictionary sensor aliases')
    groups = {entry['group'] for entry in dictionary}
    if set(sensor['groups']) - groups - {'all'}:
        raise ValueError('Unknown sensor profile group')
    include = {s.strip() for s in sensor['include']}; exclude = {s.strip() for s in sensor['exclude']}
    unknown = (include | exclude) - set(names)
    if unknown:
        raise ValueError('Unknown sensor names: ' + str(sorted(unknown)))
    candidate = lambda name: bool(re.match(r'^(Temp_|MFC|AUX|VG\d+[ _]|O2Density_|APC Valve|ValveAct_)', name))
    if any(not candidate(name) for name in include):
        raise ValueError('Process metadata cannot be selected as a sensor')
    selected = [entry['name'].strip() for entry in dictionary
        if candidate(entry['name'].strip()) and entry['name'].strip() not in exclude
        and ('all' in sensor['groups'] or entry['group'] in sensor['groups'] or entry['name'].strip() in include)]
    if not selected:
        raise ValueError('Sensor profile selects no input channels')
    targets = [name for name in selected if not name.startswith('ValveAct_')]
    if not targets:
        raise ValueError('At least one numeric sensor is required for synthetic change training')
    result.update(main_steps=steps, selected_sensors=selected, target_sensors=targets,
        selection=dict(sha256={k: sha(v) for k, v in paths.items()},
            files={k: str(v) for k, v in paths.items()},
            profiles={k: snapshots[k] for k in keys},
            ordering='dictionary order; exclude wins; Main Steps sorted',
            selected_sensors=selected, selected_steps=steps))
    return result
